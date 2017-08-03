//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2007-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_IMPL_HPP)
#define HPX_THREADMANAGER_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/state.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a thread-manager class is the central instance of management for
    /// all (non-depleted) threads
    class HPX_EXPORT threadmanager
    {
    private:
        // we use a simple mutex to protect the data members of the
        // thread manager for now
        typedef compat::mutex mutex_type;

    public:
        typedef threads::policies::callback_notifier notification_policy_type;
        typedef std::unique_ptr<detail::thread_pool_base> pool_type;
        typedef threads::policies::scheduler_base scheduler_type;
        typedef std::vector<pool_type> pool_vector;

#ifdef HPX_HAVE_TIMER_POOL
        threadmanager(util::io_service_pool& timer_pool,
                notification_policy_type& notifier);
#else
        threadmanager(notification_policy_type& notifier);
#endif
        ~threadmanager();

        void init();

        //! FIXME move to private and add --hpx:printpools cmd line option
        void print_pools(std::ostream&);

        // Get functions
        detail::thread_pool_base& default_pool() const;

        scheduler_type& default_scheduler() const;

        detail::thread_pool_base& get_pool(std::string const& pool_name) const;
        detail::thread_pool_base& get_pool(detail::pool_id_type pool_id) const;
        detail::thread_pool_base& get_pool(std::size_t thread_index) const;

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
        ///         instantiates the specified number of OS threads in each
        ///         pool. All OS threads are started to execute the function
        ///         \a tfunc.
        ///
        /// \returns      The function returns \a true if the thread manager
        ///               has been started successfully, otherwise it returns
        ///               \a false.
        bool run();

        /// \brief Forcefully stop the thread-manager
        ///
        /// \param blocking
        ///
        void stop (bool blocking = true);

        /// \brief Return whether the thread manager is still running
        //! This returns the "minimal state", i.e. the state of the
        //! least advanced thread pool
        state status() const
        {
            hpx::state result(last_valid_runtime_state);

            for (auto& pool_iter : pools_)
            {
                hpx::state s = pool_iter->get_state();
                result = (std::min)(result, s);
            }

            return result;
        }

        /// \brief return the number of HPX-threads with the given state
        ///
        /// \note This function lock the internal OS lock in the thread manager
        std::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false);

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
            std::size_t total = 0;
            for (auto& pool_iter : pools_)
            {
                total += pool_iter->get_os_thread_count();
            }
            return total;
        }

        compat::thread& get_os_thread_handle(std::size_t num_thread)
        {
            std::lock_guard<mutex_type> lk(mtx_);
            detail::pool_id_type id = threads_lookup_[num_thread];
            detail::thread_pool_base& pool = get_pool(id);
            return pool.get_os_thread_handle(num_thread);
        }

    public:
        /// API functions forwarding to notification policy

        /// This notifies the thread manager that the passed exception has been
        /// raised. The exception will be routed through the notifier and the
        /// scheduler (which will result in it being passed to the runtime
        /// object, which in turn will report it to the console, etc.).
        void report_error(std::size_t num_thread, std::exception_ptr const& e)
        {
            // propagate the error reporting to all pools, which in turn
            // will propagate to schedulers
            for (auto& pool_iter : pools_)
            {
                pool_iter->report_error(num_thread, e);
            }
        }

        // Return the (global) sequence number of the current thread
        std::size_t get_worker_thread_num(bool* numa_sensitive = nullptr)
        {
            if (get_self_ptr() == nullptr)
                return std::size_t(-1);
            return detail::thread_num_tss_.get_worker_thread_num();
        }

    public:
        /// The function register_counter_types() is called during startup to
        /// allow the registration of all performance counter types for this
        /// thread-manager instance.
        void register_counter_types();

        /// Returns the mask identifying all processing units used by this
        /// thread manager.
        mask_type get_used_processing_units() const
        {
            mask_type total_used_processing_punits = mask_type();
            threads::resize(
                total_used_processing_punits, hardware_concurrency());

            for (auto& pool_iter : pools_)
            {
                total_used_processing_punits |=
                    pool_iter->get_used_processing_units();
            }

            return total_used_processing_punits;
        }

        void set_scheduler_mode(threads::policies::scheduler_mode mode)
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->set_scheduler_mode(mode);
            }
        }

        void reset_thread_distribution()
        {
            for (auto& pool_iter : pools_)
            {
                pool_iter->reset_thread_distribution();
            }
        }

        void init_tss(std::size_t num)
        {
            detail::thread_num_tss_.init_tss(num);
        }

        void deinit_tss()
        {
            detail::thread_num_tss_.deinit_tss();
        }

    private:
        // counter creator functions
        naming::gid_type thread_counts_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);
        naming::gid_type scheduler_utilization_counter_creator(
            performance_counters::counter_info const& info, error_code& ec);

        typedef std::int64_t (threadmanager::*threadmanager_counter_func)(
            bool reset);
        typedef std::int64_t (detail::thread_pool_base::*threadpool_counter_func)(
            std::size_t num_thread, bool reset);

        naming::gid_type locality_pool_thread_counter_creator(
            threadmanager_counter_func total_func,
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        naming::gid_type queue_wait_time_counter_creator(
            threadmanager_counter_func total_func,
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);
#endif

        naming::gid_type locality_pool_thread_no_total_counter_creator(
            threadpool_counter_func pool_func,
            performance_counters::counter_info const& info, error_code& ec);

        // performance counters
        std::int64_t get_queue_length(bool reset);
#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        std::int64_t get_average_thread_wait_time(bool reset);
        std::int64_t get_average_task_wait_time(bool reset);
#endif
        std::int64_t get_cumulative_duration(bool reset);

        std::int64_t get_thread_count_unknown(bool reset)
        {
            return get_thread_count(
                unknown, thread_priority_default, std::size_t(-1), reset);
        }
        std::int64_t get_thread_count_active(bool reset)
        {
            return get_thread_count(
                active, thread_priority_default, std::size_t(-1), reset);
        }
        std::int64_t get_thread_count_pending(bool reset)
        {
            return get_thread_count(
                pending, thread_priority_default, std::size_t(-1), reset);
        }
        std::int64_t get_thread_count_suspended(bool reset)
        {
            return get_thread_count(
                suspended, thread_priority_default, std::size_t(-1), reset);
        }
        std::int64_t get_thread_count_terminated(bool reset)
        {
            return get_thread_count(
                terminated, thread_priority_default, std::size_t(-1), reset);
        }
        std::int64_t get_thread_count_staged(bool reset)
        {
            return get_thread_count(
                staged, thread_priority_default, std::size_t(-1), reset);
        }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
        std::int64_t avg_idle_rate(bool reset);
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::int64_t avg_creation_idle_rate(bool reset);
        std::int64_t avg_cleanup_idle_rate(bool reset);
#endif
#endif

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
        std::int64_t get_executed_threads(bool reset);
        std::int64_t get_executed_thread_phases(bool reset);
#ifdef HPX_HAVE_THREAD_IDLE_RATES
        std::int64_t get_thread_duration(bool reset);
        std::int64_t get_thread_phase_duration(bool reset);
        std::int64_t get_thread_overhead(bool reset);
        std::int64_t get_thread_phase_overhead(bool reset);
        std::int64_t get_cumulative_thread_duration(bool reset);
        std::int64_t get_cumulative_thread_overhead(bool reset);
#endif
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(bool reset);
        std::int64_t get_num_pending_accesses(bool reset);
        std::int64_t get_num_stolen_from_pending(bool reset);
        std::int64_t get_num_stolen_from_staged(bool reset);
        std::int64_t get_num_stolen_to_pending(bool reset);
        std::int64_t get_num_stolen_to_staged(bool reset);
#endif

private:
        mutable mutex_type mtx_; // mutex protecting the members

        // specified by the user in command line, or all cores by default
        // represents the total number of OS threads, irrespective of how many
        // are in which pool.
        std::size_t num_threads_;

        std::vector<detail::pool_id_type> threads_lookup_;

#ifdef HPX_HAVE_TIMER_POOL
        util::io_service_pool& timer_pool_;     // used for timed set_state
#endif
        pool_vector pools_;

        notification_policy_type& notifier_;

        // startup barrier
        boost::scoped_ptr<compat::barrier> startup_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
