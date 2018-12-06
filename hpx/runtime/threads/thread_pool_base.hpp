//  Copyright (c)      2018 Mikael Simberg
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/thread_pool_helpers.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/state.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    struct pool_id_type
    {
        pool_id_type(std::size_t index, std::string const& name)
          : index_(index), name_(name)
        {}

        std::size_t index() const { return index_; };
        std::string const& name() const { return name_; }

    private:
        std::size_t const index_;
        std::string const name_;
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // note: this data structure has to be protected from races from the outside

    /// \brief The base class used to manage a pool of OS threads.
    class thread_pool_base : public detail::manage_executor
    {
    public:
        /// \cond NOINTERNAL
        thread_pool_base(threads::policies::callback_notifier& notifier,
            std::size_t index, std::string const& pool_name,
            policies::scheduler_mode m, std::size_t thread_offset);

        virtual ~thread_pool_base() = default;

        virtual void init(std::size_t num_threads, std::size_t threads_offset);

        virtual bool run(std::unique_lock<compat::mutex>& l,
            std::size_t num_threads) = 0;

        virtual void stop(
            std::unique_lock<compat::mutex>& l, bool blocking = true) = 0;

        virtual void print_pool(std::ostream&) = 0;

        pool_id_type get_pool_id()
        {
            return id_;
        }
        /// \endcond

        /// Resumes the thread pool. When the all OS threads on the thread pool
        /// have been resumed the returned future will be ready.
        ///
        /// \note Can only be called from an HPX thread. Use resume_cb or
        ///       resume_direct to suspend the pool from outside HPX.
        ///
        /// \returns A `future<void>` which is ready when the thread pool has
        ///          been resumed.
        ///
        /// \throws hpx::exception if called from outside the HPX runtime.
        virtual hpx::future<void> resume() = 0;

        /// Resumes the thread pool. Takes a callback as a parameter which will
        /// be called when all OS threads on the thread pool have been resumed.
        ///
        /// \param callback [in] called when the thread pool has been resumed.
        /// \param ec       [in,out] this represents the error status on exit, if this
        ///                 is pre-initialized to \a hpx#throws the function will throw
        ///                 on error instead.
        virtual void resume_cb(
            std::function<void(void)> callback, error_code& ec = throws) = 0;

        /// Resumes the thread pool. Blocks until all OS threads on the thread pool
        /// have been resumed.
        ///
        /// \param ec [in,out] this represents the error status on exit, if this
        ///           is pre-initialized to \a hpx#throws the function will
        ///           throw on error instead.
        virtual void resume_direct(error_code& ec = throws) = 0;

        /// Suspends the thread pool. When the all OS threads on the thread pool
        /// have been suspended the returned future will be ready.
        ///
        /// \note Can only be called from an HPX thread. Use suspend_cb or
        ///       suspend_direct to suspend the pool from outside HPX. A thread
        ///       pool cannot be suspended from an HPX thread running on the
        ///       pool itself.
        ///
        /// \returns A `future<void>` which is ready when the thread pool has
        ///          been suspended.
        ///
        /// \throws hpx::exception if called from outside the HPX runtime.
        virtual hpx::future<void> suspend() = 0;

        /// Suspends the thread pool. Takes a callback as a parameter which will
        /// be called when all OS threads on the thread pool have been suspended.
        ///
        /// \note A thread pool cannot be suspended from an HPX thread running
        ///       on the pool itself.
        ///
        /// \param callback [in] called when the thread pool has been suspended.
        /// \param ec       [in,out] this represents the error status on exit, if this
        ///                 is pre-initialized to \a hpx#throws the function will throw
        ///                 on error instead.
        ///
        /// \throws hpx::exception if called from an HPX thread which is running
        ///         on the pool itself.
        virtual void suspend_cb(
            std::function<void(void)> callback, error_code& ec = throws) = 0;

        /// Suspends the thread pool. Blocks until all OS threads on the thread pool
        /// have been suspended.
        ///
        /// \note A thread pool cannot be suspended from an HPX thread running
        ///       on the pool itself.
        ///
        /// \param ec [in,out] this represents the error status on exit, if this
        ///           is pre-initialized to \a hpx#throws the function will
        ///           throw on error instead.
        ///
        /// \throws hpx::exception if called from an HPX thread which is running
        ///         on the pool itself.
        virtual void suspend_direct(error_code& ec = throws) = 0;

    public:
        /// \cond NOINTERNAL
        std::size_t get_worker_thread_num() const;
        virtual std::size_t get_os_thread_count() const = 0;

        virtual compat::thread& get_os_thread_handle(
            std::size_t num_thread) = 0;

        virtual std::size_t get_active_os_thread_count() const;

        virtual void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec) = 0;
        virtual void create_work(thread_init_data& data,
            thread_state_enum initial_state, error_code& ec) = 0;

        virtual thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec) = 0;

        virtual thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec) = 0;

        std::size_t get_pool_index() const
        {
            return id_.index();
        }
        std::string const& get_pool_name() const
        {
            return id_.name();
        }
        std::size_t get_thread_offset() const
        {
            return thread_offset_;
        }

        virtual policies::scheduler_base* get_scheduler() const
        {
            return nullptr;
        }

        mask_type get_used_processing_units() const;
        hwloc_bitmap_ptr get_numa_domain_bitmap() const;

        // performance counters
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        virtual std::int64_t get_executed_threads(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_executed_thread_phases(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        virtual std::int64_t get_thread_phase_duration(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_thread_duration(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_thread_phase_overhead(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_thread_overhead(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_cumulative_thread_duration(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_cumulative_thread_overhead(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#endif
#endif

        virtual std::int64_t get_cumulative_duration(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
        virtual std::int64_t get_background_work_duration(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_background_overhead(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        virtual std::int64_t avg_idle_rate_all(bool /*reset*/) { return 0; }
        virtual std::int64_t avg_idle_rate(std::size_t, bool) { return 0; }

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        virtual std::int64_t avg_creation_idle_rate(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t avg_cleanup_idle_rate(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#endif
#endif

        virtual std::int64_t get_queue_length(std::size_t, bool) { return 0; }

#if defined(HPX_HAVE_THREAD_QUEUE_WAITTIME)
        virtual std::int64_t get_average_thread_wait_time(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_average_task_wait_time(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#endif

#if defined(HPX_HAVE_THREAD_STEALING_COUNTS)
        virtual std::int64_t get_num_pending_misses(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_num_pending_accesses(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }

        virtual std::int64_t get_num_stolen_from_pending(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_num_stolen_to_pending(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_num_stolen_from_staged(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
        virtual std::int64_t get_num_stolen_to_staged(
            std::size_t /*thread_num*/, bool /*reset*/) { return 0; }
#endif

        virtual std::int64_t get_thread_count(thread_state_enum /*state*/,
            thread_priority /*priority*/, std::size_t /*num_thread*/,
            bool /*reset*/) { return 0; }

        virtual std::int64_t get_background_thread_count() { return 0; }

        std::int64_t get_thread_count_unknown(
            std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                unknown, thread_priority_default, num_thread, reset);
        }
        std::int64_t get_thread_count_active(std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                active, thread_priority_default, num_thread, reset);
        }
        std::int64_t get_thread_count_pending(
            std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                pending, thread_priority_default, num_thread, reset);
        }
        std::int64_t get_thread_count_suspended(
            std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                suspended, thread_priority_default, num_thread, reset);
        }
        std::int64_t get_thread_count_terminated(
            std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                terminated, thread_priority_default, num_thread, reset);
        }
        std::int64_t get_thread_count_staged(std::size_t num_thread, bool reset)
        {
            return get_thread_count(
                staged, thread_priority_default, num_thread, reset);
        }

        virtual std::int64_t get_scheduler_utilization() const = 0;

        virtual std::int64_t get_idle_loop_count(
            std::size_t num, bool reset) = 0;
        virtual std::int64_t get_busy_loop_count(
            std::size_t num, bool reset) = 0;

        ///////////////////////////////////////////////////////////////////////
        virtual bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& /*f*/,
            thread_state_enum /*state*/ = unknown) const
        {
            return false;
        }

        virtual void reset_thread_distribution() {}

        virtual void set_scheduler_mode(threads::policies::scheduler_mode) {}

        //
        virtual void abort_all_suspended_threads() {}
        virtual bool cleanup_terminated(bool /*delete_all*/) { return false; }

        virtual hpx::state get_state() const = 0;
        virtual hpx::state get_state(std::size_t num_thread) const = 0;

        virtual bool has_reached_state(hpx::state s) const = 0;

        virtual void do_some_work(std::size_t /*num_thread*/) {}

        virtual void report_error(std::size_t num, std::exception_ptr const& e)
        {
            notifier_.on_error(num, e);
        }

        ///////////////////////////////////////////////////////////////////////
        // detail::manage_executor implementation

        /// \brief Return the requested policy element.
        virtual std::size_t get_policy_element(detail::executor_parameter p,
            error_code& ec = throws) const = 0;

        // \brief Return statistics collected by this scheduler.
        virtual void get_statistics(executor_statistics& stats,
            error_code& ec = throws) const = 0;

        // \brief Provide the given processing unit to the scheduler.
        virtual void add_processing_unit(std::size_t virt_core,
            std::size_t thread_num, error_code& ec = throws) = 0;

        // \brief Remove the given processing unit from the scheduler.
        virtual void remove_processing_unit(std::size_t thread_num,
            error_code& ec = throws) = 0;

        // \brief Return the description string of the underlying scheduler.
        char const* get_description() const;

        /// \endcond

        /// Suspends the given processing unit. When the processing unit has
        /// been suspended the returned future will be ready.
        ///
        /// \note Can only be called from an HPX thread. Use
        ///       suspend_processing_unit_cb or to suspend the processing unit
        ///       from outside HPX. Requires that the pool has
        ///       threads::policies::enable_elasticity set.
        ///
        /// \param virt_core [in] The processing unit on the the pool to be
        ///                  suspended. The processing units are indexed
        ///                  starting from 0.
        ///
        /// \returns A `future<void>` which is ready when the given processing
        ///          unit has been suspended.
        ///
        /// \throws hpx::exception if called from outside the HPX runtime.
        virtual hpx::future<void> suspend_processing_unit(std::size_t virt_core) = 0;

        /// Suspends the given processing unit. Takes a callback as a parameter
        /// which will be called when the processing unit has been suspended.
        ///
        /// \note Requires that the pool has
        ///       threads::policies::enable_elasticity set.
        ///
        /// \param callback  [in] Callback which is called when the processing
        ///                  unit has been suspended.
        /// \param virt_core [in] The processing unit to suspend.
        /// \param ec        [in,out] this represents the error status on exit, if this
        ///                  is pre-initialized to \a hpx#throws the function will throw
        ///                  on error instead.
        virtual void suspend_processing_unit_cb(
            std::function<void(void)> callback, std::size_t virt_core,
            error_code& ec = throws) = 0;

        /// Resumes the given processing unit. When the processing unit has been
        /// resumed the returned future will be ready.
        ///
        /// \note Can only be called from an HPX thread. Use
        ///       resume_processing_unit_cb or to resume the processing unit
        ///       from outside HPX. Requires that the pool has
        ///       threads::policies::enable_elasticity set.
        ///
        /// \param virt_core [in] The processing unit on the the pool to be
        ///                  resumed. The processing units are indexed starting
        ///                  from 0.
        ///
        /// \returns A `future<void>` which is ready when the given processing
        ///          unit has been resumed.
        virtual hpx::future<void> resume_processing_unit(std::size_t virt_core) = 0;

        /// Resumes the given processing unit. Takes a callback as a parameter
        /// which will be called when the processing unit has been resumed.
        ///
        /// \note Requires that the pool has
        ///       threads::policies::enable_elasticity set.
        ///
        /// \param callback  [in] Callback which is called when the processing
        ///                  unit has been suspended.
        /// \param virt_core [in] The processing unit to resume.
        /// \param ec        [in,out] this represents the error status on exit, if this
        ///                  is pre-initialized to \a hpx#throws the function will throw
        ///                  on error instead.
        virtual void resume_processing_unit_cb(
            std::function<void(void)> callback, std::size_t virt_core,
                error_code& ec = throws) = 0;

        /// \cond NOINTERNAL
        policies::scheduler_mode get_scheduler_mode() const
        {
            return mode_;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        void init_pool_time_scale();
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        pool_id_type id_;

        // Mode of operation of the pool
        policies::scheduler_mode mode_;

        // The thread_offset is equal to the accumulated number of
        // threads in all pools preceding this pool
        // in the thread indexation. That means, that in order to know
        // the global index of a thread it owns, the pool has to compute:
        // global index = thread_offset_ + local index.
        std::size_t thread_offset_;

        // scale timestamps to nanoseconds
        double timestamp_scale_;

        // callback functions to invoke at start, stop, and error
        threads::policies::callback_notifier& notifier_;
        /// \endcond
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
