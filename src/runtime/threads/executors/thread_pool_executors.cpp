//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/runtime/threads/resource_manager.hpp>
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executors.hpp>
#include <hpx/util/register_locks.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    void get_processing_units(std::size_t num_threads,
        std::vector<std::size_t>& punits)
    {
        for (std::size_t i = 0; i != num_threads; ++i)
            punits.push_back(i);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    class manage_thread_pool_executor
      : public threads::detail::manage_executor
    {
    public:
        manage_thread_pool_executor(thread_pool_executor<Scheduler>& sched)
          : sched_(sched)
        {}

    protected:
        // Return the requested policy element.
        std::size_t get_policy_element(threads::detail::executor_parameter p,
            error_code& ec) const
        {
            return sched_.get_policy_element(p, ec);
        }

        // Return statistics collected by this scheduler
        void get_statistics(executor_statistics& stats, error_code& ec) const
        {
            sched_.get_statistics(stats, ec);
        }

        // Provide the given processing unit to the scheduler.
        void add_processing_unit(std::size_t virt_core, std::size_t thread_num, 
            error_code& ec)
        {
            sched_.add_processing_unit(virt_core, thread_num, ec);
        }

        // Remove the given processing unit from the scheduler.
        void remove_processing_unit(std::size_t thread_num, error_code& ec)
        {
            sched_.remove_processing_unit(thread_num, ec);
        }

    private:
        thread_pool_executor<Scheduler>& sched_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool_executor<Scheduler>::thread_pool_executor(std::size_t max_punits,
            std::size_t min_punits)
      : scheduler_(max_punits), shutdown_sem_(0),
        states_(max_punits), puinits_(max_punits),
        current_concurrency_(0), tasks_scheduled_(0), tasks_completed_(0),
        max_punits_(max_punits), min_punits_(min_punits), cookie_(0)
    {
        states_.resize(max_punits);
        for (std::size_t i = 0; i != max_punits; ++i)
            states_[i].store(starting);

        // Inform the resource manager about this new executor. This causes the
        // resource manager to interact with this executor using the
        // manage_executor interface.
        resource_manager& rm = resource_manager::get();
        cookie_ = rm.initial_allocation(new manage_thread_pool_executor<Scheduler>(*this));
    }

    template <typename Scheduler>
    thread_pool_executor<Scheduler>::~thread_pool_executor()
    {
        // Inform the resource manager that this executor is about to be 
        // destroyed. This will cause it to invoke remove_processing_unit below
        // for each of the currently allocated virtual cores.
        resource_manager& rm = resource_manager::get();
        rm.stop_executor(cookie_);

        // wait for executor to finish executing
        shutdown_sem_.wait();

        // detach this executor from resource manager
        rm.detach(cookie_);     // this releases proxy (manage_thread_pool_executor)

#if defined(HPX_DEBUG)
        // all resources should have been stopped at this point
        for (std::size_t i = 0; i != states_.size(); ++i)
        {
            BOOST_ASSERT(states_[i].load() == stopping);
        }

        // all scheduled tasks should have completed executing
        BOOST_ASSERT(tasks_completed_ == tasks_scheduled_);

        // all driver threads should have stopped executing
        BOOST_ASSERT(current_concurrency_ == 0);
#endif
    }

    template <typename Scheduler>
    threads::thread_state_enum 
    thread_pool_executor<Scheduler>::thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        // execute the actual thread function
        func();

        // update statistics
        ++tasks_completed_;

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add(
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, error_code& ec)
    {
        // create a new thread
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this,
            boost::move(f)), desc);

        // update statistics
        ++tasks_scheduled_;

        threads::detail::create_thread(scheduler_, data, initial_state, run_now, ec);
        if (ec) {
            --tasks_scheduled_;
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add_at(
        boost::posix_time::ptime const& abs_time,
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* desc, 
        error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this,
            boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended, true, ec);
        if (ec) return;
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // update statistics
        ++tasks_scheduled_;

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, abs_time, id, ec);
        if (ec) {
            --tasks_scheduled_;
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add_after(
        boost::posix_time::time_duration const& rel_time,
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* desc, 
        error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(util::bind(
            &thread_pool_executor::thread_function_nullary, this,
            boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended, true, ec);
        if (ec) return;
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // update statistics
        ++tasks_scheduled_;

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, rel_time, id, ec);
        if (ec) {
            --tasks_scheduled_;
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    std::size_t thread_pool_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();
        return scheduler_.get_queue_length();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct on_run_exit
    {
        on_run_exit(boost::atomic<std::size_t>& current_concurrency,
                lcos::local::counting_semaphore& shutdown_sem)
          : current_concurrency_(current_concurrency),
            shutdown_sem_(shutdown_sem)
        {}

        ~on_run_exit()
        {
            --current_concurrency_;
            shutdown_sem_.signal();
        }

        boost::atomic<std::size_t>& current_concurrency_;
        lcos::local::counting_semaphore& shutdown_sem_;
    };

    // execute all work
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::run(std::size_t virt_core)
    {
        // Set the state to 'running' only if it's still in 'starting' state, 
        // otherwise our destructor is currently being executed, which means
        // we need to still execute all threads.
        state expected = starting;
        states_[virt_core].compare_exchange_strong(expected, running);

        ++current_concurrency_;

        {
            on_run_exit on_exit(current_concurrency_, shutdown_sem_);

            boost::int64_t executed_threads = 0;
            boost::uint64_t overall_times = 0, thread_times = 0;
            threads::detail::scheduling_loop(virt_core, scheduler_, 
                states_[virt_core], executed_threads, overall_times, thread_times);
        }
    }

    // Return statistics collected by this scheduler
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::get_statistics(
        executor_statistics& stats, error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        stats.queue_length_ = scheduler_.get_queue_length();
        stats.tasks_scheduled_ = tasks_scheduled_;
        stats.tasks_completed_ = tasks_completed_;
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t thread_pool_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
        case threads::detail::min_concurrency:
            return min_punits_;

        case threads::detail::max_concurrency:
            return max_punits_;

        case threads::detail::current_concurrency:
            return current_concurrency_;

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    // Provide the given processing unit to the scheduler.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add_processing_unit(
        std::size_t virt_core, std::size_t thread_num, error_code& ec)
    {
        register_thread_nullary(
            util::bind(&thread_pool_executor::run, this, virt_core),
            "thread_pool_executor thread", threads::pending, true,
            threads::thread_priority_normal, thread_num,
            threads::thread_stacksize_default, ec);
    }

    // Remove the given processing unit from the scheduler.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::remove_processing_unit(
        std::size_t virt_core, error_code& ec)
    {
        // inform the scheduler to stop the virtual core
        states_[virt_core].store(stopping);
    }
}}}}

namespace hpx { namespace threads { namespace executors
{
    ///////////////////////////////////////////////////////////////////////////
    local_queue_executor::local_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : executor(new detail::thread_pool_executor<
            policies::local_queue_scheduler<lcos::local::spinlock> >(
                max_punits, min_punits))
    {}

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_executor::local_priority_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : executor(new detail::thread_pool_executor<
            policies::local_priority_queue_scheduler<lcos::local::spinlock> >(
                max_punits, min_punits))
    {}

#if defined(HPX_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_executor::static_priority_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : executor(new detail::thread_pool_executor<
            policies::static_priority_queue_scheduler<lcos::local::spinlock> >(
                max_punits, min_punits))
    {}
#endif
}}}
