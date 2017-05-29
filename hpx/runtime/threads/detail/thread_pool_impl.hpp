//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREAD_POOL_HPP)
#define HPX_THREAD_POOL_HPP

#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>

namespace hpx { namespace threads { namespace detail
{

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper;


    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    class thread_pool_impl : public thread_pool
    {
    public:

        //! Constructor used in constructor of threadmanager
        thread_pool_impl(
            Scheduler* sched,
            threads::policies::callback_notifier& notifier,
            std::size_t index, char const* pool_name,
            policies::scheduler_mode m = policies::nothing_special);

        //! Constructor used in constructor of thread_pool_os_executor
        thread_pool_impl(
            Scheduler& sched,
            threads::policies::callback_notifier& notifier,
            std::size_t index, char const* pool_name,
            policies::scheduler_mode m = policies::nothing_special);

        ~thread_pool_impl();

        void print_pool();

        hpx::state get_state() const;
        hpx::state get_state(std::size_t num_thread) const;
        bool has_reached_state(hpx::state s) const;
        void init(std::size_t num_threads, std::size_t threads_offset);
        void init(std::size_t num_threads, std::size_t threads_offset,
                          policies::init_affinity_data const& data);

        std::size_t get_pu_num(std::size_t num_thread) const;

        mask_cref_type get_pu_mask(topology const& topology, std::size_t num_thread) const;


        void do_some_work(std::size_t num_thread);
        void report_error(std::size_t num, boost::exception_ptr const& e);

        void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec);

        void create_work(thread_init_data& data, thread_state_enum initial_state, error_code& ec);

        thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec);

        void abort_all_suspended_threads();

        bool cleanup_terminated(bool delete_all);

        std::int64_t get_thread_count(
                thread_state_enum state, thread_priority priority,
                std::size_t num, bool reset) const;

        bool enumerate_threads(util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state) const;

        void reset_thread_distribution();

        void set_scheduler_mode(threads::policies::scheduler_mode mode);
        bool run(std::unique_lock<compat::mutex>& l, std::size_t num_threads, std::size_t thread_offset);

        //! used to be templated over the lock. Now not bc can't have templated virtual functions ...
        void stop_locked(std::unique_lock<lcos::local::no_mutex>& l, bool blocking = true);
        void stop_locked(std::unique_lock<compat::mutex>& l, bool blocking = true);

        std::size_t get_os_thread_count() const
        {
            return threads_.size();
        }

        compat::thread& get_os_thread_handle(std::size_t num_thread);

        void thread_func(std::size_t num_thread, topology const& topology, compat::barrier& startup);

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
       std::int64_t avg_creation_idle_rate(bool reset);
       std::int64_t avg_cleanup_idle_rate(bool reset);
#endif
#endif

        std::int64_t get_queue_length(std::size_t num_thread) const;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        std::int64_t get_average_thread_wait_time(std::size_t num_thread) const;
        std::int64_t get_average_task_wait_time(std::size_t num_thread) const;
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(std::size_t num, bool reset);

        std::int64_t get_num_pending_accesses(std::size_t num, bool reset);

        std::int64_t get_num_stolen_from_pending(std::size_t num, bool reset);

        std::int64_t get_num_stolen_to_pending(std::size_t num, bool reset);

        std::int64_t get_num_stolen_from_staged(std::size_t num, bool reset);

        std::int64_t get_num_stolen_to_staged(std::size_t num, bool reset);
#endif


    protected:

        std::vector<compat::thread> threads_; //! vector of OS-threads

        friend struct init_tss_helper<Scheduler>;


    private:
        // hold the used scheduler
        Scheduler& sched_;//! previously: reference to


    };



} // namespace detail
} // namespace threads
} // namespace hpx

#endif
