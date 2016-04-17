//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
#endif
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
#endif
#include <hpx/runtime/threads/executors/thread_pool_os_executors.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>

#include <boost/atomic.hpp>

#include <mutex>
#include <string>

namespace hpx
{
    threads::policies::callback_notifier
        get_notification_policy(char const* prefix);
}

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::string thread_pool_os_executor<Scheduler>::get_unique_name()
    {
        std::string name = Scheduler::get_scheduler_name();
        name += "#";
        name += std::to_string(++os_executor_count_);
        return name;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    boost::atomic<std::size_t>
        thread_pool_os_executor<Scheduler>::os_executor_count_(0);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool_os_executor<Scheduler>::thread_pool_os_executor(
            std::size_t num_punits, std::string const& affinity_desc)
      : scheduler_(num_punits),
        executor_name_(get_unique_name()),
        notifier_(get_notification_policy(executor_name_.c_str())),
        pool_(scheduler_, notifier_, executor_name_.c_str()),
        num_threads_(num_punits)
    {
        if (num_punits > hpx::threads::hardware_concurrency())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "thread_pool_os_executor<Scheduler>::thread_pool_os_executor",
                "max_punit shouldn't be larger than number of available "
                "OS-threads");
            return;
        }

        std::unique_lock<mutex_type> lk(mtx_);

        // initialize the affinity configuration for this scheduler
        threads::policies::init_affinity_data data("pu", affinity_desc);
        pool_.init(num_threads_, data);

        if (!pool_.run(lk, num_threads_))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "thread_pool_os_executor<Scheduler>::thread_pool_os_executor",
                "couldn't start thread_pool");
        }
    }

    template <typename Scheduler>
    thread_pool_os_executor<Scheduler>::~thread_pool_os_executor()
    {
        // if we're still starting up, give this executor a chance of executing
        // its tasks
        while (!scheduler_.has_reached_state(state_running))
            this_thread::suspend();

        // inform the scheduler to stop the core
        {
            std::unique_lock<mutex_type> lk(mtx_);
            pool_.stop(lk, true);
        }

#if defined(HPX_DEBUG)
        // all resources should have been stopped at this point (or have never
        // been initialized)
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
            hpx::state s = scheduler_.get_state(i).load();
            HPX_ASSERT(s == state_initialized || s == state_stopped);
        }
//
//         // all scheduled tasks should have completed executing
//         HPX_ASSERT(tasks_completed_ == tasks_scheduled_);
//
//         // all driver threads should have stopped executing
//         HPX_ASSERT(current_concurrency_ == 0);
#endif
    }

    template <typename Scheduler>
    threads::thread_state_enum
    thread_pool_os_executor<Scheduler>::thread_function_nullary(
        closure_type func)
    {
        // execute the actual thread function
        func();

        return threads::terminated;
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t thread_pool_os_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
        case threads::detail::min_concurrency:
        case threads::detail::max_concurrency:
        case threads::detail::current_concurrency:
            return num_threads_;

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_os_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::add(closure_type && f,
        util::thread_description const& desc,
        threads::thread_state_enum initial_state,
        bool run_now, threads::thread_stacksize stacksize, error_code& ec)
    {
        // create a new thread
        thread_init_data data(util::bind(
            util::one_shot(&thread_pool_os_executor::thread_function_nullary),
            std::move(f)), desc);
        data.stacksize = threads::get_stack_size(stacksize);

        threads::thread_id_type id = threads::invalid_thread_id;
        pool_.create_thread(data, id, initial_state, run_now, ec);
        if (ec) return;

        HPX_ASSERT(invalid_thread_id != id || !run_now);

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::add_at(
        boost::chrono::steady_clock::time_point const& abs_time,
        closure_type && f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(util::bind(
            util::one_shot(&thread_pool_os_executor::thread_function_nullary),
            std::move(f)), desc);
        data.stacksize = threads::get_stack_size(stacksize);

        threads::thread_id_type id = threads::invalid_thread_id;
        pool_.create_thread(data, id, suspended, true, ec);
        if (ec) return;

        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        pool_.set_state(abs_time, id, pending, wait_timeout,
            thread_priority_normal, ec);
        if (ec) return;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::add_after(
        boost::chrono::steady_clock::duration const& rel_time,
        closure_type && f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(boost::chrono::steady_clock::now() + rel_time,
            std::move(f), desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    boost::uint64_t thread_pool_os_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        std::lock_guard<mutex_type> lk(mtx_);
        return pool_.get_thread_count(unknown, thread_priority_default,
            std::size_t(-1), false);
    }

    // Reset internal (round robin) thread distribution scheme
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::reset_thread_distribution()
    {
        pool_.reset_thread_distribution();
    }
}}}}

namespace hpx { namespace threads { namespace executors
{
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    local_queue_os_executor::local_queue_os_executor()
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_queue_scheduler<> >(get_os_thread_count()))
    {}

    local_queue_os_executor::local_queue_os_executor(
            std::size_t num_threads, std::string const& affinity_desc)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_queue_scheduler<> >(num_threads, affinity_desc))
    {}
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_queue_os_executor::static_queue_os_executor()
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_queue_scheduler<> >(get_os_thread_count()))
    {}

    static_queue_os_executor::static_queue_os_executor(
            std::size_t num_threads, std::string const& affinity_desc)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_queue_scheduler<> >(num_threads, affinity_desc))
    {}
#endif

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_os_executor::local_priority_queue_os_executor()
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_priority_queue_scheduler<> >(get_os_thread_count()))
    {}

    local_priority_queue_os_executor::local_priority_queue_os_executor(
            std::size_t num_threads, std::string const& affinity_desc)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_priority_queue_scheduler<> >(
                num_threads, affinity_desc))
    {}

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_os_executor::static_priority_queue_os_executor()
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_priority_queue_scheduler<> >(get_os_thread_count()))
    {}

    static_priority_queue_os_executor::static_priority_queue_os_executor(
            std::size_t num_threads, std::string const& affinity_desc)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_priority_queue_scheduler<> >(
                num_threads, affinity_desc))
    {}
#endif
}}}
