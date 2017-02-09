//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2007-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_IMPL_HPP)
#define HPX_THREADMANAGER_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/state.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/thread/mutex.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a thread-manager class is the central instance of management for
    /// all (non-depleted) threads
    template <typename SchedulingPolicy>
    class HPX_EXPORT threadmanager_impl : public threadmanager_base
    {
    private:
        // we use a simple mutex to protect the data members of the
        // thread manager for now
        typedef boost::mutex mutex_type;

    public:
        typedef SchedulingPolicy scheduling_policy_type;
        typedef threads::policies::callback_notifier notification_policy_type;


        ///
        threadmanager_impl(util::io_service_pool& timer_pool,
            scheduling_policy_type& scheduler,
            notification_policy_type& notifier,
            std::size_t num_threads);
        ~threadmanager_impl();

        std::size_t init(policies::init_affinity_data const& data);

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
        void register_work(thread_init_data& data,
            thread_state_enum initial_state = pending,
            error_code& ec = throws);

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
        void register_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state = pending,
            bool run_now = true, error_code& ec = throws);

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
        bool run(std::size_t num_threads = 1);

        /// \brief Forcefully stop the thread-manager
        ///
        /// \param blocking
        ///
        void stop (bool blocking = true);

        /// \brief Return whether the thread manager is still running
        state status() const
        {
            return pool_.get_state();
        }

        /// \brief return the number of HPX-threads with the given state
        ///
        /// \note This function lock the internal OS lock in the thread manager
        std::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const;

        // Enumerate all matching threads
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const;

        // \brief Abort all threads which are in suspended state. This will set
        //        the state of all suspended threads to \a pending while
        //        supplying the wait_abort extended state flag
        void abort_all_suspended_threads();

        // \brief Clean up terminated threads. This deletes all threads which
        //        have been terminated but which are still held in the queue
        //        of terminated threads. Some schedulers might not do anything
        //        here.
        bool cleanup_terminated(bool delete_all = false);

        /// \brief Return the number of OS threads running in this thread-manager
        ///
        /// This function will return correct results only if the thread-manager
        /// is running.
        std::size_t get_os_thread_count() const
        {
            std::lock_guard<mutex_type> lk(mtx_);
            return pool_.get_os_thread_count();
        }

        boost::thread& get_os_thread_handle(std::size_t num_thread)
        {
            std::lock_guard<mutex_type> lk(mtx_);
            return pool_.get_os_thread_handle(num_thread);
        }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
        /// Get percent maintenance time in main thread-manager loop.
        std::int64_t avg_idle_rate(bool reset);
        std::int64_t avg_idle_rate(std::size_t num_thread, bool reset);
#endif
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::int64_t avg_creation_idle_rate(bool reset);
        std::int64_t avg_cleanup_idle_rate(bool reset);
#endif

    public:
        /// this notifies the thread manager that there is some more work
        /// available
        void do_some_work(std::size_t num_thread = std::size_t(-1))
        {
            pool_.do_some_work(num_thread);
        }

        /// API functions forwarding to notification policy
        void report_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            pool_.report_error(num_thread, e);
        }

        std::size_t get_worker_thread_num(bool* numa_sensitive = nullptr)
        {
            if (get_self_ptr() == nullptr)
                return std::size_t(-1);
            return pool_.get_worker_thread_num();
        }

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
        std::int64_t get_executed_threads(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_executed_thread_phases(
            std::size_t num = std::size_t(-1), bool reset = false);

#ifdef HPX_HAVE_THREAD_IDLE_RATES
        std::int64_t get_thread_phase_duration(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_thread_duration(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_thread_phase_overhead(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_thread_overhead(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_cumulative_thread_duration(
            std::size_t num = std::size_t(-1), bool reset = false);
        std::int64_t get_cumulative_thread_overhead(
            std::size_t num = std::size_t(-1), bool reset = false);
#endif
#endif

        std::int64_t get_cumulative_duration(
            std::size_t num = std::size_t(-1), bool reset = false);

    protected:
        ///
        template <typename C>
        void start_periodic_maintenance(std::true_type);

        template <typename C>
        void start_periodic_maintenance(std::false_type) {}

        template <typename C>
        void periodic_maintenance_handler(std::true_type);

        template <typename C>
        void periodic_maintenance_handler(std::false_type) {}

    public:
        /// The function register_counter_types() is called during startup to
        /// allow the registration of all performance counter types for this
        /// thread-manager instance.
        void register_counter_types();

        /// Returns of the number of the processing units the given thread
        /// is allowed to run on
        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return pool_.get_pu_num(num_thread);
        }

        /// Return the mask for processing units the given thread is allowed
        /// to run on.
        mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread) const
        {
            return pool_.get_pu_mask(topology, num_thread);
        }

        // Returns the mask identifying all processing units used by this
        // thread manager.
        mask_cref_type get_used_processing_units() const
        {
            return pool_.get_used_processing_units();
        }

        void set_scheduler_mode(threads::policies::scheduler_mode mode)
        {
            pool_.set_scheduler_mode(mode);
        }

        void reset_thread_distribution()
        {
            pool_.reset_thread_distribution();
        }

    private:
        // counter creator functions
        naming::gid_type queue_length_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
        naming::gid_type thread_counts_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
#ifdef HPX_HAVE_THREAD_IDLE_RATES
        naming::gid_type idle_rate_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
#endif
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        naming::gid_type thread_wait_time_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
        naming::gid_type task_wait_time_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
#endif

        naming::gid_type scheduler_utilization_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);

        naming::gid_type idle_loop_count_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
        naming::gid_type busy_loop_count_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);

    private:
        mutable mutex_type mtx_;   // mutex protecting the members

        std::size_t num_threads_;
        util::io_service_pool& timer_pool_;     // used for timed set_state

        util::block_profiler<register_thread_tag> thread_logger_;
        util::block_profiler<register_work_tag> work_logger_;
        util::block_profiler<set_state_tag> set_state_logger_;

        detail::thread_pool<scheduling_policy_type> pool_;
        notification_policy_type& notifier_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
