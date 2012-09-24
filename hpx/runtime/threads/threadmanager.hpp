//  Copyright (c) 2007-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_MAY_20_2008_845AM)
#define HPX_THREADMANAGER_MAY_20_2008_845AM

#include <hpx/config.hpp>

#include <boost/cstdint.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/posix_time_config.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
/*
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/spinlock.hpp>
*/
#include <hpx/util/thread_specific_ptr.hpp>

#include <hpx/config/warnings_prefix.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    struct register_thread_tag {};
    struct register_work_tag {};
    struct set_state_tag {};

    struct thread_init_data;

    ///////////////////////////////////////////////////////////////////////////
    struct threadmanager_base : private boost::noncopyable
    {
    protected:
        // we use the boost::posix_time::ptime type for time representation
        typedef boost::posix_time::ptime time_type;

        // we use the boost::posix_time::time_duration type as the duration
        // representation
        typedef boost::posix_time::time_duration duration_type;

    public:
        virtual ~threadmanager_base() {}

        /// \brief Return whether the thread manager is still running
        virtual state status() const = 0;

        /// \brief return the number of PX-threads with the given state
        virtual boost::int64_t get_thread_count(
            thread_state_enum state = unknown) const = 0;

        // \brief Abort all threads which are in suspended state. This will set
        //        the state of all suspended threads to \a pending while
        //        supplying the wait_abort extended state flag
        virtual void abort_all_suspended_threads() = 0;

        // \brief Clean up terminated threads. This deletes all threads which
        //        have been terminated but which are still held in the queue
        //        of terminated threads. Some schedulers might not do anything
        //        here.
        virtual bool cleanup_terminated() = 0;

        /// The get_phase function is part of the thread related API. It
        /// queries the phase of one of the threads known to the threadmanager
        ///
        /// \param id       [in] The thread id of the thread the phase should
        ///                 be returned for.
        ///
        /// \returns        This function returns the current phase of the
        ///                 thread referenced by the \a id parameter. If the
        ///                 thread is not known to the threadmanager the return
        ///                 value will be ~0.
        virtual std::size_t get_phase(thread_id_type id) = 0;

        /// The get_state function is part of the thread related API. It
        /// queries the state of one of the threads known to the threadmanager
        ///
        /// \param id       [in] The thread id of the thread the state should
        ///                 be returned for.
        ///
        /// \returns        This function returns the current state of the
        ///                 thread referenced by the \a id parameter. It will
        ///                 return one of the values as defined by the
        ///                 \a thread_state enumeration. If the
        ///                 thread is not known to the threadmanager the return
        ///                 value will be \a thread_state#unknown.
        virtual thread_state get_state(thread_id_type id) = 0;

        /// The set_state function is part of the thread related API and allows
        /// to change the state of one of the threads managed by this
        /// threadmanager.
        ///
        /// \param id       [in] The thread id of the thread the state should
        ///                 be modified for.
        /// \param newstate [in] The new state to be set for the thread
        ///                 referenced by the \a id parameter.
        /// \param newstate_ex [in] The new extended state to be set for the
        ///                 thread referenced by the \a id parameter.
        /// \param priority   [in] The priority with which the thread will be
        ///                   executed if the parameter \a newstate is pending.
        ///
        /// \returns        This function returns the previous state of the
        ///                 thread referenced by the \a id parameter. It will
        ///                 return one of the values as defined by the
        ///                 \a thread_state enumeration. If the
        ///                 thread is not known to the threadmanager the return
        ///                 value will be \a thread_state#unknown.
        ///
        /// \note           If the thread referenced by the parameter \a id
        ///                 is in \a thread_state#active state this function
        ///                 schedules a new thread which will set the state of
        ///                 the thread as soon as its not active anymore. The
        ///                 function returns \a thread_state#active in this case.
        virtual thread_state set_state(thread_id_type id,
            thread_state_enum newstate,
            thread_state_ex_enum newstate_ex = wait_signaled,
            thread_priority priority = thread_priority_normal,
            error_code& ec = throws) = 0;

        /// Set a timer to set the state of the given \a thread to the given
        /// new value after it expired (at the given time)
        /// \brief  Set the thread state of the \a thread referenced by the
        ///         thread_id \a id.
        ///
        /// Set a timer to set the state of the given \a thread to the given
        /// new value after it expired (at the given time)
        ///
        /// \param id         [in] The thread id of the thread the state should
        ///                   be modified for.
        /// \param at_time
        /// \param state      [in] The new state to be set for the thread
        ///                   referenced by the \a id parameter.
        /// \param newstate_ex [in] The new extended state to be set for the
        ///                   thread referenced by the \a id parameter.
        /// \param priority   [in] The priority with which the thread will be
        ///                   executed if the parameter \a newstate is pending.
        ///
        /// \returns
        virtual thread_id_type set_state (time_type const& expire_at,
            thread_id_type id, thread_state_enum newstate = pending,
            thread_state_ex_enum newstate_ex = wait_timeout,
            thread_priority priority = thread_priority_normal,
            error_code& ec = throws) = 0;

        /// \brief  Set the thread state of the \a thread referenced by the
        ///         thread_id \a id.
        ///
        /// Set a timer to set the state of the given \a thread to the given
        /// new value after it expired (after the given duration)
        ///
        /// \param id         [in] The thread id of the thread the state should
        ///                   be modified for.
        /// \param after_duration
        /// \param state      [in] The new state to be set for the thread
        ///                   referenced by the \a id parameter.
        /// \param newstate_ex [in] The new extended state to be set for the
        ///                   thread referenced by the \a id parameter.
        /// \param priority   [in] The priority with which the thread will be
        ///                   executed if the parameter \a newstate is pending.
        ///
        /// \returns
        virtual thread_id_type set_state (duration_type const& expire_from_now,
            thread_id_type id, thread_state_enum newstate = pending,
            thread_state_ex_enum newstate_ex = wait_timeout,
            thread_priority priority = thread_priority_normal,
            error_code& ec = throws) = 0;

        /// The get_description function is part of the thread related API and
        /// allows to query the description of one of the threads known to the
        /// threadmanager
        ///
        /// \param id       [in] The thread id of the thread the description
        ///                 should be returned for.
        ///
        /// \returns        This function returns the description of the
        ///                 thread referenced by the \a id parameter. If the
        ///                 thread is not known to the threadmanager the return
        ///                 value will be the string "<unknown>".
        virtual char const* get_description(thread_id_type id) const = 0;
        virtual char const* set_description(thread_id_type id, char const* desc = 0) = 0;

        virtual char const* get_lco_description(thread_id_type id) const = 0;
        virtual char const* set_lco_description(thread_id_type id, char const* desc = 0) = 0;

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
        ///
        /// \returns      The function returns the thread id of the newly
        ///               created thread.
        virtual thread_id_type
        register_thread(thread_init_data& data,
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

        /// The get_interruption_enabled function is part of the thread related
        /// API. It queries whether of one of the threads known can be
        /// interrupted.
        ///
        /// \param id       [in] The thread id of the thread to query.
        ///
        /// \returns        This function returns whether the thread referenced
        ///                 by the \a id parameter can be interrupted. If the
        ///                 thread is not known to the thread-manager the return
        ///                 value will be false.
        virtual bool get_interruption_enabled(thread_id_type id,
            error_code& ec = throws) = 0;

        /// The set_interruption_enabled function is part of the thread related
        /// API. It sets whether of one of the threads known can be
        /// interrupted.
        ///
        /// \param id       [in] The thread id of the thread to query.
        virtual bool set_interruption_enabled(thread_id_type id, bool enable,
            error_code& ec = throws) = 0;

        /// The get_interruption_requested function is part of the thread related
        /// API. It queries whether of one of the threads known has been
        /// interrupted.
        ///
        /// \param id       [in] The thread id of the thread to query.
        ///
        /// \returns        This function returns whether the thread referenced
        ///                 by the \a id parameter has been interrupted. If the
        ///                 thread is not known to the thread-manager the return
        ///                 value will be false.
        virtual bool get_interruption_requested(thread_id_type id,
            error_code& ec = throws) = 0;

        /// The interrupt function is part of the thread related API. It
        /// queries notifies one of the threads to abort at the next
        /// interruption point.
        ///
        /// \param id       [in] The thread id of the thread to interrupt.
        virtual void interrupt(thread_id_type id, error_code& ec = throws) = 0;

        /// The run_thread_exit_callbacks function is part of the thread related
        /// API. It runs all exit functions for one of the threads.
        ///
        /// \param id       [in] The thread id of the thread to interrupt.
        virtual void run_thread_exit_callbacks(thread_id_type id,
            error_code& ec = throws) = 0;

        /// The add_thread_exit_callback function is part of the thread related
        /// API. It adds a callback function to be executed at thread exit.
        ///
        /// \param id       [in] The thread id of the thread to interrupt.
        ///
        /// \returns        This function returns whether the function has been
        ///                 added to the referenced thread
        ///                 by the \a id parameter has been interrupted. If the
        ///                 thread is not known to the thread-manager the return
        ///                 value will be false.
        virtual bool add_thread_exit_callback(thread_id_type id,
            HPX_STD_FUNCTION<void()> const& f, error_code& ec = throws) = 0;

        ///
        virtual void free_thread_exit_callbacks(thread_id_type id,
            error_code& ec = throws) = 0;

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

        /// Return number of the processing unit the given thread is running on
        virtual std::size_t get_pu_num(std::size_t num_thread) = 0;
        
        virtual boost::int64_t get_executed_threads(std::size_t num = std::size_t(-1)) const = 0;

        static std::size_t get_worker_thread_num(bool* numa_sensitive = 0);

        void init_tss(std::size_t thread_num, bool numa_sensitive);
        void deinit_tss();

    private:
        // the TSS holds the number associated with a given OS thread
        struct tls_tag {};
        static hpx::util::thread_specific_ptr<std::size_t, tls_tag> thread_num_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
