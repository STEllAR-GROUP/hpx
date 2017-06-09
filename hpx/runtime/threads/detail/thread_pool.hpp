//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM)
#define HPX_RUNTIME_THREADS_DETAIL_THREAD_POOL_JUN_11_2015_1137AM

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/state.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util_fwd.hpp>

#include <boost/atomic.hpp>
#include <boost/scoped_ptr.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail
{

    ///////////////////////////////////////////////////////////////////////////
    struct manage_active_thread_count
    {
        manage_active_thread_count(boost::atomic<long>& counter)
                : counter_(counter)
        {
            ++counter_;
        }
        ~manage_active_thread_count()
        {
            --counter_;
        }

        boost::atomic<long>& counter_;
    };


    ///////////////////////////////////////////////////////////////////////////
    struct pool_id_type
    {
        pool_id_type(std::size_t index, std::string name)
                : index_(index), name_(name)
        {}

        std::size_t index_;
        std::string name_;
        //! could get an hpx::naming::id_type in the future
    };

    ///////////////////////////////////////////////////////////////////////////
    // note: this data structure has to be protected from races from the outside
    class HPX_EXPORT thread_pool
    {
    public:
        thread_pool(
            threads::policies::callback_notifier& notifier, std::size_t index,
            char const* pool_name, policies::scheduler_mode m = policies::nothing_special);
        ~thread_pool();

        virtual void print_pool() = 0;

        pool_id_type get_pool_id(){
            return id_;
        }

        virtual void init(std::size_t num_threads, std::size_t threads_offset) = 0;
        virtual void init(std::size_t num_threads, std::size_t threads_offset,
                          policies::detail::affinity_data const& data) = 0;

        virtual bool run(std::unique_lock<compat::mutex>& l, std::size_t num_threads, std::size_t thread_offset) = 0;
        void stop(std::unique_lock<compat::mutex>& l, bool blocking = true);

        virtual void stop_locked(std::unique_lock<lcos::local::no_mutex>& l, bool blocking = true) = 0;
        virtual void stop_locked(std::unique_lock<compat::mutex>& l, bool blocking = true) = 0;

        std::size_t get_worker_thread_num() const;
        virtual std::size_t get_os_thread_count() const = 0;

        virtual compat::thread& get_os_thread_handle(std::size_t num_thread) = 0;

        virtual void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec) = 0;
        virtual void create_work(thread_init_data& data,
            thread_state_enum initial_state, error_code& ec) = 0;

        thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec);

        virtual thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec) = 0;

        const std::string &get_pool_name() const
        {
            return id_.name_;
        }

        virtual policies::scheduler_base *get_scheduler() const = 0;

        virtual std::size_t get_pu_num(std::size_t num_thread) const = 0;
        virtual mask_cref_type get_pu_mask(std::size_t num_thread) const = 0;
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
        virtual std::int64_t avg_creation_idle_rate(bool reset) = 0;
        virtual std::int64_t avg_cleanup_idle_rate(bool reset) = 0;
#endif
#endif

        virtual std::int64_t get_queue_length(std::size_t num_thread) const = 0;

#if defined(HPX_HAVE_THREAD_QUEUE_WAITTIME)
        std::int64_t get_average_thread_wait_time(
            std::size_t num_thread) const = 0;
        std::int64_t get_average_task_wait_time(
            std::size_t num_thread) const = 0;
#endif

#if defined(HPX_HAVE_THREAD_STEALING_COUNTS)
        virtual std::int64_t get_num_pending_misses(std::size_t num, bool reset) = 0;
        virtual std::int64_t get_num_pending_accesses(std::size_t num, bool reset) = 0;

        virtual std::int64_t get_num_stolen_from_pending(std::size_t num, bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_pending(std::size_t num, bool reset) = 0;
        virtual std::int64_t get_num_stolen_from_staged(std::size_t num, bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_staged(std::size_t num, bool reset) = 0;
#endif

        virtual std::int64_t get_thread_count(thread_state_enum state,
            thread_priority priority, std::size_t num_thread, bool reset) const = 0;

        std::int64_t get_scheduler_utilization() const;

        std::int64_t get_idle_loop_count(std::size_t num) const;
        std::int64_t get_busy_loop_count(std::size_t num) const;

        ///////////////////////////////////////////////////////////////////////
        virtual bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const = 0;

        virtual void reset_thread_distribution() = 0;

        virtual void set_scheduler_mode(threads::policies::scheduler_mode mode)= 0;

        //
        virtual void abort_all_suspended_threads() = 0;
        virtual bool cleanup_terminated(bool delete_all) = 0;

        virtual hpx::state get_state() const = 0;
        virtual hpx::state get_state(std::size_t num_thread) const = 0;

        virtual bool has_reached_state(hpx::state s) const = 0;

        virtual void do_some_work(std::size_t num_thread) = 0;

        virtual void report_error(std::size_t num, std::exception_ptr const& e) = 0;

    protected:

        void init_tss(std::size_t num);
        void deinit_tss();

        virtual  void thread_func(std::size_t num_thread, topology const& topology,
            compat::barrier& startup) = 0;

        double timestamp_scale_;    // scale timestamps to nanoseconds

//! should I leave them here or move them to thread_pool_impl?
//    private:
        threads::policies::callback_notifier& notifier_;
        pool_id_type id_;

        // startup barrier
        boost::scoped_ptr<compat::barrier> startup_;

        // count number of executed HPX-threads and thread phases (invocations)
        std::vector<std::int64_t> executed_threads_;
        std::vector<std::int64_t> executed_thread_phases_;
        boost::atomic<long> thread_count_;

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

        std::vector<std::int64_t> idle_loop_counts_, busy_loop_counts_;

        std::vector<std::uint8_t> tasks_active_;

        // Stores the mask identifying all processing units used by this
        // thread manager.
        threads::mask_type used_processing_units_;

        // Mode of operation of the pool
        policies::scheduler_mode mode_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
