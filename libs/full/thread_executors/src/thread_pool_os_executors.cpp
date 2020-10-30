//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_POOL_OS_EXECUTOR_COMPATIBILITY)
#include <hpx/thread_executors/thread_pool_os_executors.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/schedulers/local_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/schedulers/static_queue_scheduler.hpp>
#endif
#include <hpx/schedulers/local_priority_queue_scheduler.hpp>
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/schedulers/static_priority_queue_scheduler.hpp>
#endif
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace hpx {
    threads::policies::callback_notifier get_notification_policy(
        char const* prefix);
}    // namespace hpx

namespace hpx { namespace threads { namespace executors { namespace detail {
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
    std::atomic<std::size_t>
        thread_pool_os_executor<Scheduler>::os_executor_count_(0);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool_os_executor<Scheduler>::thread_pool_os_executor(
        std::size_t num_threads,
        policies::detail::affinity_data const& affinity_data,
        util::optional<policies::callback_notifier> notifier)
      : scheduler_(nullptr)
      , executor_name_(get_unique_name())
      , notifier_(notifier.has_value() ?
                notifier.value() :
                get_notification_policy(executor_name_.c_str()))
      , pool_(nullptr)
      , network_background_callback_()
      , thread_pool_init_(executor_name_, 0,
            policies::scheduler_mode::default_mode, num_threads, 0, notifier_,
            affinity_data, network_background_callback_)
    {
        if (num_threads > hpx::threads::hardware_concurrency())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "thread_pool_os_executor<Scheduler>::thread_pool_os_executor",
                "max_punit shouldn't be larger than number of available "
                "OS-threads");
            return;
        }

        typename Scheduler::init_parameter_type init(
            num_threads, thread_pool_init_.affinity_data_);
        std::unique_ptr<Scheduler> scheduler(new Scheduler(init));
        scheduler_ = scheduler.get();

        pool_.reset(new threads::detail::scheduled_thread_pool<Scheduler>(
            std::move(scheduler), thread_pool_init_));

        std::unique_lock<mutex_type> lk(mtx_);
        if (!pool_->run(lk, num_threads))
        {
            lk.unlock();
            HPX_THROW_EXCEPTION(invalid_status,
                "thread_pool_os_executor<Scheduler>::thread_pool_os_executor",
                "couldn't start thread_pool");
        }
    }

    template <typename Scheduler>
    thread_pool_os_executor<Scheduler>::~thread_pool_os_executor()
    {
        //  if we're still starting up, give this executor a chance of executing
        // its tasks
        hpx::util::yield_while(
            [this]() { return !scheduler_->has_reached_state(state_running); });

        // inform the scheduler to stop the core
        {
            std::unique_lock<mutex_type> lk(mtx_);
            pool_->stop(lk, true);
        }

#if defined(HPX_DEBUG)
        // all resources should have been stopped at this point (or have never
        // been initialized)
        for (std::size_t i = 0; i != thread_pool_init_.num_threads_; ++i)
        {
            hpx::state s = scheduler_->get_state(i).load();
            HPX_ASSERT(s == state_initialized || s == state_stopped);
        }
#endif
    }

    template <typename Scheduler>
    threads::thread_result_type
    thread_pool_os_executor<Scheduler>::thread_function_nullary(
        closure_type func)
    {
        // execute the actual thread function
        func();

        return threads::thread_result_type(
            threads::thread_schedule_state::terminated,
            threads::invalid_thread_id);
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t thread_pool_os_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch (p)
        {
        case threads::detail::min_concurrency:
        case threads::detail::max_concurrency:
        case threads::detail::current_concurrency:
            return thread_pool_init_.num_threads_;

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
    void thread_pool_os_executor<Scheduler>::add(closure_type&& f,
        util::thread_description const& desc,
        threads::thread_schedule_state initial_state, bool run_now,
        threads::thread_stacksize stacksize,
        threads::thread_schedule_hint /* schedulehint */, error_code& ec)
    {
        // create a new thread
        thread_init_data data(
            util::one_shot(
                util::bind(&thread_pool_os_executor::thread_function_nullary,
                    std::move(f))),
            desc, thread_priority::default_, thread_schedule_hint(), stacksize,
            initial_state, run_now);

        threads::thread_id_type id = threads::invalid_thread_id;
        pool_->create_thread(data, id, ec);
        if (ec)
            return;

        HPX_ASSERT(invalid_thread_id != id || !run_now);

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::add_at(
        std::chrono::steady_clock::time_point const& abs_time, closure_type&& f,
        util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(
            util::one_shot(
                util::bind(&thread_pool_os_executor::thread_function_nullary,
                    std::move(f))),
            desc, thread_priority::default_, thread_schedule_hint(), stacksize,
            thread_schedule_state::suspended, true);

        threads::thread_id_type id = threads::invalid_thread_id;
        pool_->create_thread(data, id, ec);
        if (ec)
            return;

        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        pool_->set_state(abs_time, id, thread_schedule_state::pending,
            thread_restart_state::timeout, thread_priority::normal, ec);
        if (ec)
            return;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::add_after(
        std::chrono::steady_clock::duration const& rel_time, closure_type&& f,
        util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(std::chrono::steady_clock::now() + rel_time, std::move(f),
            desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    std::uint64_t thread_pool_os_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        std::lock_guard<mutex_type> lk(mtx_);
        return pool_->get_thread_count(thread_schedule_state::unknown,
            thread_priority::default_, std::size_t(-1), false);
    }

    // Reset internal (round robin) thread distribution scheme
    template <typename Scheduler>
    void thread_pool_os_executor<Scheduler>::reset_thread_distribution()
    {
        pool_->reset_thread_distribution();
    }
}}}}    // namespace hpx::threads::executors::detail

namespace hpx { namespace threads { namespace executors {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    local_queue_os_executor::local_queue_os_executor(std::size_t num_threads,
        policies::detail::affinity_data const& affinity_data,
        util::optional<policies::callback_notifier> notifier)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_queue_scheduler<>>(
            num_threads, affinity_data, notifier))
    {
    }
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_queue_os_executor::static_queue_os_executor(std::size_t num_threads,
        policies::detail::affinity_data const& affinity_data,
        util::optional<policies::callback_notifier> notifier)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_queue_scheduler<>>(
            num_threads, affinity_data, notifier))
    {
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_os_executor::local_priority_queue_os_executor(
        std::size_t num_threads,
        policies::detail::affinity_data const& affinity_data,
        util::optional<policies::callback_notifier> notifier)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::local_priority_queue_scheduler<>>(
            num_threads, affinity_data, notifier))
    {
    }

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_os_executor::static_priority_queue_os_executor(
        std::size_t num_threads,
        policies::detail::affinity_data const& affinity_data,
        util::optional<policies::callback_notifier> notifier)
      : scheduled_executor(new detail::thread_pool_os_executor<
            policies::static_priority_queue_scheduler<>>(
            num_threads, affinity_data, notifier))
    {
    }
#endif
}}}    // namespace hpx::threads::executors
#endif
