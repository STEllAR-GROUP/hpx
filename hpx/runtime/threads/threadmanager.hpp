//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2007-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <hpx/config.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/runtime/threads/executors/current_executor.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/state.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util_fwd.hpp>

#include <boost/exception_ptr.hpp>

#include <cstddef>
#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    struct register_thread_tag {};
    struct register_work_tag {};
    struct set_state_tag {};

    class thread_init_data;

    ///////////////////////////////////////////////////////////////////////////
    struct threadmanager_base
    {
    private:
        HPX_NON_COPYABLE(threadmanager_base);

    public:
        threadmanager_base() {}

        virtual ~threadmanager_base() {}

        virtual std::size_t init(policies::init_affinity_data const& data) = 0;

        /// \brief Return whether the thread manager is still running
        virtual state status() const = 0;

        /// \brief return the number of HPX-threads with the given state
        virtual std::int64_t get_thread_count(
            thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1),
            bool reset = false) const = 0;

        // Enumerate all matching threads
        virtual bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const = 0;

        // \brief Abort all threads which are in suspended state. This will set
        //        the state of all suspended threads to \a pending while
        //        supplying the wait_abort extended state flag
        virtual void abort_all_suspended_threads() = 0;

        // \brief Clean up terminated threads. This deletes all threads which
        //        have been terminated but which are still held in the queue
        //        of terminated threads. Some schedulers might not do anything
        //        here.
        virtual bool cleanup_terminated(bool delete_all = false) = 0;

        /// The function \a register_work adds a new work item to the thread
        /// manager. It doesn't immediately create a new \a thread, it just adds
        /// the task parameters (function, initial state and description) to
        /// the internal management data structures. The thread itself will be
        /// created when the number of existing threads drops below the number
        /// of threads specified by the constructors max_count parameter.
        ///
        /// \param func   [in] The function or function object to execute as
        ///               the thread's function. This must have a signature as
        ///               defined by \a thread_function_type.
        /// \param description [in] The value of this parameter allows to
        ///               specify a description of the thread to create. This
        ///               information is used for logging purposes mainly, but
        ///               might be useful for debugging as well. This parameter
        ///               is optional and defaults to an empty string.
        /// \param initial_state
        ///               [in] The value of this parameter defines the initial
        ///               state of the newly created \a thread. This must be
        ///               one of the values as defined by the \a thread_state
        ///               enumeration (thread_state#pending, or \a
        ///               thread_state#suspended, any other value will throw a
        ///               hpx#bad_parameter exception).
        virtual void
        register_work(thread_init_data& data,
            thread_state_enum initial_state = pending,
            error_code& ec = throws) = 0;

        /// The function \a register_thread adds a new work item to the thread
        /// manager. It creates a new \a thread, adds it to the internal
        /// management data structures, and schedules the new thread, if
        /// appropriate.
        ///
        /// \param func   [in] The function or function object to execute as
        ///               the thread's function. This must have a signature as
        ///               defined by \a thread_function_type.
        /// \param id     [out] This parameter will hold the id of the created
        ///               thread. This id is guaranteed to be validly
        ///               initialized before the thread function is executed.
        /// \param description [in] The value of this parameter allows to
        ///               specify a description of the thread to create. This
        ///               information is used for logging purposes mainly, but
        ///               might be useful for debugging as well. This parameter
        ///               is optional and defaults to an empty string.
        /// \param initial_state
        ///               [in] The value of this parameter defines the initial
        ///               state of the newly created \a thread. This must be
        ///               one of the values as defined by the \a thread_state
        ///               enumeration (thread_state#pending, or \a
        ///               thread_state#suspended, any other value will throw a
        ///               hpx#bad_parameter exception).
        /// \param run_now [in] If this parameter is \a true and the initial
        ///               state is given as \a thread_state#pending the thread
        ///               will be run immediately, otherwise it will be
        ///               scheduled to run later (either this function is
        ///               called for another thread using \a true for the
        ///               parameter \a run_now or the function \a
        ///               threadmanager#do_some_work is called). This parameter
        ///               is optional and defaults to \a true.
        virtual void
        register_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state = pending,
            bool run_now = true, error_code& ec = throws) = 0;

        /// \brief  Run the thread manager's work queue. This function
        ///         instantiates the specified number of OS threads. All OS
        ///         threads are started to execute the function \a tfunc.
        ///
        /// \param num_threads
        ///               [in] The initial number of threads to be started by
        ///               this thread manager instance. This parameter is
        ///               optional and defaults to 1 (one).
        ///
        /// \returns      The function returns \a true if the thread manager
        ///               has been started successfully, otherwise it returns
        ///               \a false.
        virtual bool run(std::size_t num_threads = 1) = 0;

        /// \brief Forcefully stop the thread-manager
        ///
        /// \param blocking
        ///
        virtual void stop (bool blocking = true) = 0;

        /// this notifies the thread manager that there is some more work
        /// available
        virtual void do_some_work(std::size_t num_thread = std::size_t(-1)) = 0;

        /// This notifies the thread manager that the passed exception has been
        /// raised. The exception will be routed through the notifier and the
        /// scheduler (which will result in it being passed to the runtime
        /// object, which in turn will report it to the console, etc.).
        virtual void report_error(std::size_t, boost::exception_ptr const&) = 0;

        /// The function register_counter_types() is called during startup to
        /// allow the registration of all performance counter types for this
        /// thread-manager instance.
        virtual void register_counter_types() = 0;

        /// Returns of the number of the processing unit the given thread
        /// is allowed to run on
        virtual std::size_t get_pu_num(std::size_t) const = 0;

        virtual compat::thread & get_os_thread_handle(std::size_t) = 0;

        /// Return the mask for processing units the given thread is allowed
        /// to run on.
        virtual mask_cref_type get_pu_mask(topology const&, std::size_t) const = 0;

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        virtual std::int64_t get_executed_threads(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;
        virtual std::int64_t get_executed_thread_phases(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;

#ifdef HPX_HAVE_THREAD_IDLE_RATES
        virtual std::int64_t get_thread_phase_duration(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;
        virtual std::int64_t get_thread_duration(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;
        virtual std::int64_t get_thread_phase_overhead(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;
        virtual std::int64_t get_thread_overhead(
            std::size_t num = std::size_t(-1), bool reset = false) = 0;
#endif
#endif

        // Returns the mask identifying all processing units used by this
        // thread manager.
        virtual mask_cref_type get_used_processing_units() const = 0;

        ///////////////////////////////////////////////////////////////////////
        virtual std::size_t get_worker_thread_num(
            bool* numa_sensitive = nullptr) = 0;

        virtual void reset_thread_distribution() = 0;

        virtual void set_scheduler_mode(threads::policies::scheduler_mode m) = 0;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
