//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SCHEDULED_THREAD_POOL_HPP)
#define HPX_SCHEDULED_THREAD_POOL_HPP

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/util/assert.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    class scheduled_thread_pool : public hpx::threads::thread_pool_base
    {
    public:
        ///////////////////////////////////////////////////////////////////
        scheduled_thread_pool(std::unique_ptr<Scheduler> sched,
            threads::policies::callback_notifier& notifier, std::size_t index,
            std::string const& pool_name, policies::scheduler_mode m =
                policies::scheduler_mode::nothing_special,
            std::size_t thread_offset = 0);
        virtual ~scheduled_thread_pool();

        void print_pool(std::ostream& os);

        threads::policies::scheduler_base* get_scheduler() const
        {
            return sched_.get();
        }

        ///////////////////////////////////////////////////////////////////
        hpx::state get_state() const;
        hpx::state get_state(std::size_t num_thread) const;

        bool has_reached_state(hpx::state s) const
        {
            return sched_->Scheduler::has_reached_state(s);
        }

        ///////////////////////////////////////////////////////////////////
        void do_some_work(std::size_t num_thread)
        {
            sched_->Scheduler::do_some_work(num_thread);
        }

        void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec);

        void create_work(thread_init_data& data,
            thread_state_enum initial_state, error_code& ec);

        thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec);

        thread_id_type set_state(
            util::steady_time_point const& abs_time, thread_id_type const& id,
            thread_state_enum newstate, thread_state_ex_enum newstate_ex,
            thread_priority priority, error_code& ec);

        void report_error(std::size_t num, std::exception_ptr const& e);

        void abort_all_suspended_threads()
        {
            sched_->Scheduler::abort_all_suspended_threads();
        }

        bool cleanup_terminated(bool delete_all)
        {
            return sched_->Scheduler::cleanup_terminated(delete_all);
        }

        std::int64_t get_thread_count(thread_state_enum state,
            thread_priority priority, std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_thread_count(
                state, priority, num, reset);
        }

        std::int64_t get_background_thread_count()
        {
            return sched_->Scheduler::get_background_thread_count();
        }

        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state) const
        {
            return sched_->Scheduler::enumerate_threads(f, state);
        }

        void reset_thread_distribution()
        {
            return sched_->Scheduler::reset_thread_distribution();
        }

        void set_scheduler_mode(threads::policies::scheduler_mode mode)
        {
            mode_ = mode;
            return sched_->Scheduler::set_scheduler_mode(mode);
        }

        ///////////////////////////////////////////////////////////////////
        bool run(std::unique_lock<compat::mutex>& l, std::size_t pool_threads);

        template <typename Lock>
        void stop_locked(Lock& l, bool blocking = true);
        void stop(std::unique_lock<compat::mutex>& l, bool blocking = true);

        hpx::future<void> suspend();
        void suspend_cb(std::function<void(void)> callback, error_code& ec = throws);
        void suspend_direct(error_code& ec = throws);

        hpx::future<void> resume();
        void resume_cb(std::function<void(void)> callback, error_code& ec = throws);
        void resume_direct(error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////
        compat::thread& get_os_thread_handle(std::size_t global_thread_num)
        {
            std::size_t num_thread_local =
                global_thread_num - this->thread_offset_;
            HPX_ASSERT(num_thread_local < threads_.size());
            return threads_[num_thread_local];
        }

        void thread_func(std::size_t thread_num, std::size_t global_thread_num,
            std::shared_ptr<compat::barrier> startup);

        std::size_t get_os_thread_count() const
        {
            return thread_count_;
        }

        std::size_t get_active_os_thread_count() const
        {
            std::size_t active_os_thread_count = 0;
            for (std::size_t thread_num = 0; thread_num < threads_.size();
                ++thread_num)
            {
                if (sched_->Scheduler::get_state(thread_num).load() ==
                    state_running)
                {
                    ++active_os_thread_count;
                }
            }

            return active_os_thread_count;
        }

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_pending_misses(num, reset);
        }

        std::int64_t get_num_pending_accesses(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_pending_accesses(num, reset);
        }

        std::int64_t get_num_stolen_from_pending(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_stolen_from_pending(num, reset);
        }

        std::int64_t get_num_stolen_to_pending(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_stolen_to_pending(num, reset);
        }

        std::int64_t get_num_stolen_from_staged(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_stolen_from_staged(num, reset);
        }

        std::int64_t get_num_stolen_to_staged(std::size_t num, bool reset)
        {
            return sched_->Scheduler::get_num_stolen_to_staged(num, reset);
        }
#endif
        std::int64_t get_queue_length(std::size_t num_thread, bool reset)
        {
            return sched_->Scheduler::get_queue_length(num_thread);
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        std::int64_t get_average_thread_wait_time(
            std::size_t num_thread, bool reset)
        {
            return sched_->Scheduler::get_average_thread_wait_time(num_thread);
        }

        std::int64_t get_average_task_wait_time(
            std::size_t num_thread, bool reset)
        {
            return sched_->Scheduler::get_average_task_wait_time(num_thread);
        }
#endif

        std::int64_t get_executed_threads() const;

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        std::int64_t get_executed_threads(std::size_t, bool);
        std::int64_t get_executed_thread_phases(std::size_t, bool);
#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::int64_t get_thread_phase_duration(std::size_t, bool);
        std::int64_t get_thread_duration(std::size_t, bool);
        std::int64_t get_thread_phase_overhead(std::size_t, bool);
        std::int64_t get_thread_overhead(std::size_t, bool);
        std::int64_t get_cumulative_thread_duration(std::size_t, bool);
        std::int64_t get_cumulative_thread_overhead(std::size_t, bool);
#endif
#endif

        std::int64_t get_cumulative_duration(std::size_t, bool);

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::int64_t get_background_work_duration(std::size_t, bool);
        std::int64_t get_background_overhead(std::size_t, bool);
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::int64_t avg_idle_rate_all(bool reset);
        std::int64_t avg_idle_rate(std::size_t, bool);

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        std::int64_t avg_creation_idle_rate(std::size_t, bool);
        std::int64_t avg_cleanup_idle_rate(std::size_t, bool);
#endif
#endif

        std::int64_t get_idle_loop_count(std::size_t num, bool reset);
        std::int64_t get_busy_loop_count(std::size_t num, bool reset);
        std::int64_t get_scheduler_utilization() const;

        ///////////////////////////////////////////////////////////////////////
        // detail::manage_executor implementation

        // Return the requested policy element
        std::size_t get_policy_element(executor_parameter p, error_code&) const;

        // Return statistics collected by this scheduler
        void get_statistics(executor_statistics& s, error_code&) const;

        // Provide the given processing unit to the scheduler.
        void add_processing_unit(std::size_t virt_core,
            std::size_t thread_num, error_code&  = hpx::throws);

        // Remove the given processing unit from the scheduler.
        void remove_processing_unit(
            std::size_t virt_core, error_code& = hpx::throws);

        // Suspend the given processing unit on the scheduler.
        hpx::future<void> suspend_processing_unit(std::size_t virt_core);
        void suspend_processing_unit_cb(std::function<void(void)> callback,
            std::size_t virt_core, error_code& = hpx::throws);

        // Resume the given processing unit on the scheduler.
        hpx::future<void> resume_processing_unit(std::size_t virt_core);
        void resume_processing_unit_cb(std::function<void(void)> callback,
            std::size_t virt_core, error_code& = hpx::throws);

    protected:
        friend struct init_tss_helper<Scheduler>;

        void resume_internal(bool blocking, error_code& ec);
        void suspend_internal(error_code& ec);

        void remove_processing_unit_internal(
            std::size_t virt_core, error_code& = hpx::throws);
        void add_processing_unit_internal(std::size_t virt_core,
            std::size_t thread_num, std::shared_ptr<compat::barrier> startup,
            error_code& ec = hpx::throws);

        void suspend_processing_unit_internal(std::size_t virt_core,
            error_code& = hpx::throws);
        void resume_processing_unit_internal(std::size_t virt_core,
            error_code& = hpx::throws);

    private:
        std::vector<compat::thread> threads_;           // vector of OS-threads

        // hold the used scheduler
        std::unique_ptr<Scheduler> sched_;

    public:
        void init_perf_counter_data(std::size_t pool_threads);

    private:
        // count number of executed HPX-threads and thread phases (invocations)
        std::vector<std::int64_t> executed_threads_;
        std::vector<std::int64_t> executed_thread_phases_;

        // scheduler utilization data
        std::vector<std::uint8_t> tasks_active_;

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        // timestamps/values of last reset operation for various performance
        // counters
        std::vector<std::int64_t> reset_executed_threads_;
        std::vector<std::int64_t> reset_executed_thread_phases_;

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::vector<std::int64_t> reset_thread_duration_;
        std::vector<std::uint64_t> reset_thread_duration_times_;

        std::vector<std::int64_t> reset_thread_overhead_;
        std::vector<std::uint64_t> reset_thread_overhead_times_;
        std::vector<std::uint64_t> reset_thread_overhead_times_total_;

        std::vector<std::int64_t> reset_thread_phase_duration_;
        std::vector<std::uint64_t> reset_thread_phase_duration_times_;

        std::vector<std::int64_t> reset_thread_phase_overhead_;
        std::vector<std::uint64_t> reset_thread_phase_overhead_times_;
        std::vector<std::uint64_t> reset_thread_phase_overhead_times_total_;

        std::vector<std::uint64_t> reset_cumulative_thread_duration_;

        std::vector<std::uint64_t> reset_cumulative_thread_overhead_;
        std::vector<std::uint64_t> reset_cumulative_thread_overhead_total_;
#endif
#endif

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::vector<std::uint64_t> reset_idle_rate_time_;
        std::vector<std::uint64_t> reset_idle_rate_time_total_;

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        std::vector<std::uint64_t> reset_creation_idle_rate_time_;
        std::vector<std::uint64_t> reset_creation_idle_rate_time_total_;

        std::vector<std::uint64_t> reset_cleanup_idle_rate_time_;
        std::vector<std::uint64_t> reset_cleanup_idle_rate_time_total_;
#endif
#endif

        // tfunc_impl timers
        std::vector<std::uint64_t> exec_times_, tfunc_times_;
        std::vector<std::uint64_t> reset_tfunc_times_;

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::vector<std::uint64_t> background_duration_;
        std::vector<std::uint64_t> reset_background_duration_;
        std::vector<std::uint64_t> reset_background_tfunc_times_;
        std::vector<std::uint64_t> reset_background_overhead_;
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

        std::vector<std::int64_t> idle_loop_counts_, busy_loop_counts_;

        // support detail::manage_executor interface
        std::atomic<long> thread_count_;
        std::atomic<std::int64_t> tasks_scheduled_;
    };
}}}    // namespace hpx::threads::detail

#include <hpx/config/warnings_suffix.hpp>

#endif
