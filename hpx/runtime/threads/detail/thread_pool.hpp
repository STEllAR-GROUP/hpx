//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/date_time_chrono.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper;

    ///////////////////////////////////////////////////////////////////////////
    // note: this data structure has to be protected from races from the outside
    template <typename Scheduler>
    class thread_pool
    {
    public:
        thread_pool(Scheduler& sched,
            threads::policies::callback_notifier& notifier, char const* pool_name,
            policies::scheduler_mode m = policies::nothing_special);
        ~thread_pool();

        std::size_t init(std::size_t num_threads,
            policies::init_affinity_data const& data);

        bool run(std::unique_lock<boost::mutex>& l, std::size_t num_threads);
        void stop(std::unique_lock<boost::mutex>& l, bool blocking = true);
        template <typename Lock>
        void stop_locked(Lock& l, bool blocking = true);

        std::size_t get_worker_thread_num() const;
        std::size_t get_os_thread_count() const
        {
            return threads_.size();
        }
        boost::thread& get_os_thread_handle(std::size_t num_thread);

        void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec);
        void create_work(thread_init_data& data,
            thread_state_enum initial_state, error_code& ec);

        thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec);

        thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec);

        std::size_t get_pu_num(std::size_t num_thread) const;
        mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread) const;
        mask_cref_type get_used_processing_units() const;

        // performance counters
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        std::int64_t get_executed_threads(std::size_t num, bool reset);
        std::int64_t get_executed_thread_phases(std::size_t num, bool reset);
#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::int64_t get_thread_phase_duration(std::size_t num, bool reset);
        std::int64_t get_thread_duration(std::size_t num, bool reset);
        std::int64_t get_thread_phase_overhead(std::size_t num, bool reset);
        std::int64_t get_thread_overhead(std::size_t num, bool reset);
        std::int64_t get_cumulative_thread_duration(std::size_t num, bool reset);
        std::int64_t get_cumulative_thread_overhead(std::size_t num, bool reset);
#endif
#endif

        std::int64_t get_cumulative_duration(std::size_t num, bool reset);

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        ///////////////////////////////////////////////////////////////////////
        std::int64_t avg_idle_rate(bool reset);
        std::int64_t avg_idle_rate(std::size_t num_thread, bool reset);

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        std::int64_t avg_creation_idle_rate(bool reset);
        std::int64_t avg_cleanup_idle_rate(bool reset);
#endif
#endif

        std::int64_t get_queue_length(std::size_t num_thread) const;

#if defined(HPX_HAVE_THREAD_QUEUE_WAITTIME)
        std::int64_t get_average_thread_wait_time(
            std::size_t num_thread) const;
        std::int64_t get_average_task_wait_time(
            std::size_t num_thread) const;
#endif

#if defined(HPX_HAVE_THREAD_STEALING_COUNTS)
        std::int64_t get_num_pending_misses(std::size_t num, bool reset);
        std::int64_t get_num_pending_accesses(std::size_t num, bool reset);

        std::int64_t get_num_stolen_from_pending(std::size_t num, bool reset);
        std::int64_t get_num_stolen_to_pending(std::size_t num, bool reset);
        std::int64_t get_num_stolen_from_staged(std::size_t num, bool reset);
        std::int64_t get_num_stolen_to_staged(std::size_t num, bool reset);
#endif

        std::int64_t get_thread_count(thread_state_enum state,
            thread_priority priority, std::size_t num_thread, bool reset) const;

        void reset_thread_distribution();

        void set_scheduler_mode(threads::policies::scheduler_mode mode);

        //
        void abort_all_suspended_threads();
        bool cleanup_terminated(bool delete_all);

        hpx::state get_state() const;
        hpx::state get_state(std::size_t num_thread) const;

        bool has_reached_state(hpx::state s) const;

        void do_some_work(std::size_t num_thread);

        void report_error(std::size_t num, boost::exception_ptr const& e);

    protected:
        friend struct init_tss_helper<Scheduler>;

        void init_tss(std::size_t num);
        void deinit_tss();

        void thread_func(std::size_t num_thread, topology const& topology,
            boost::barrier& startup);

    private:
        // this thread manager has exactly as many OS-threads as requested
        std::vector<boost::thread> threads_;

        // refer to used scheduler
        Scheduler& sched_;
        threads::policies::callback_notifier& notifier_;
        std::string pool_name_;

        // startup barrier
        boost::scoped_ptr<boost::barrier> startup_;

        // count number of executed HPX-threads and thread phases (invocations)
        std::vector<std::int64_t> executed_threads_;
        std::vector<std::int64_t> executed_thread_phases_;
        boost::atomic<long> thread_count_;

        double timestamp_scale_;    // scale timestamps to nanoseconds

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

        // Stores the mask identifying all processing units used by this
        // thread manager.
        threads::mask_type used_processing_units_;

        // Mode of operation of the pool
        policies::scheduler_mode mode_;
    };
}}}

#endif
