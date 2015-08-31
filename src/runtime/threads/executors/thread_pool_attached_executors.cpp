//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/threads/threadmanager_impl.hpp>

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
#include <hpx/runtime/threads/executors/thread_pool_attached_executors.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/thread/locks.hpp>

namespace hpx
{
    threads::policies::callback_notifier
        get_notification_policy(char const* prefix);
}

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool_attached_executor<Scheduler>::thread_pool_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority, thread_stacksize stacksize)
      : first_thread_(first_thread),
        num_threads_(num_threads),
        os_thread_(0),
        priority_(priority),
        stacksize_(stacksize),
        thread_manager_(
            static_cast<threadmanager_impl<Scheduler> &>(
                get_thread_manager()
            )
        )
    {
//         if (first_thread + num_threads > hpx::get_os_thread_count())
//         {
//             HPX_THROW_EXCEPTION(bad_parameter,
//                 "thread_pool_attached_executor<Scheduler>::"
//                     "thread_pool_attached_executor",
//                 "first_thread + num_threads shouldn't be larger than number of "
//                 "available OS-threads");
//             return;
//         }
    }

    template <typename Scheduler>
    threads::thread_state_enum
    thread_pool_attached_executor<Scheduler>::thread_function_nullary(
        closure_type func)
    {
        // execute the actual thread function
        func();

        return threads::terminated;
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    template <typename Scheduler>
    void thread_pool_attached_executor<Scheduler>::add(closure_type && f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, threads::thread_stacksize stacksize, error_code& ec)
    {
        if (stacksize == threads::thread_stacksize_default)
            stacksize = stacksize_;

//         register_thread_nullary(std::move(f), desc, initial_state, run_now,
//             priority_, get_next_thread_num(), stacksize, ec);

        // create a new thread
        thread_init_data data(util::bind(
            util::one_shot(&thread_pool_attached_executor::thread_function_nullary),
            std::move(f)), desc);
        data.stacksize = threads::get_stack_size(stacksize);
        data.num_os_thread = get_next_thread_num();

        threads::thread_id_type id = threads::invalid_thread_id;
        thread_manager_.register_thread(data, id, initial_state, true, ec);
        if (ec) return;

        HPX_ASSERT(invalid_thread_id != id || !run_now);

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_attached_executor<Scheduler>::add_at(
        boost::chrono::steady_clock::time_point const& abs_time,
        closure_type && f, char const* description,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        if (stacksize == threads::thread_stacksize_default)
            stacksize = stacksize_;

        // create new thread
        thread_id_type id = register_thread_nullary(
            std::move(f), description, suspended, false,
            priority_, get_next_thread_num(), stacksize, ec);
        if (ec) return;

        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        set_thread_state(id, abs_time);
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    template <typename Scheduler>
    void thread_pool_attached_executor<Scheduler>::add_after(
        boost::chrono::steady_clock::duration const& rel_time,
        closure_type && f, char const* description,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(boost::chrono::steady_clock::now() + rel_time,
            std::move(f), description, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    boost::uint64_t thread_pool_attached_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_count() - get_thread_count(terminated);
    }

    // Reset internal (round robin) thread distribution scheme
    template <typename Scheduler>
    void thread_pool_attached_executor<Scheduler>::reset_thread_distribution()
    {
        os_thread_.store(0);        // start over from first thread
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t thread_pool_attached_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
        case threads::detail::min_concurrency:
        case threads::detail::max_concurrency:
        case threads::detail::current_concurrency:
            return hpx::get_os_thread_count();

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_attached_executor<Scheduler>::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }
}}}}

namespace hpx { namespace threads { namespace executors
{
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    local_queue_attached_executor::local_queue_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority, thread_stacksize stacksize)
      : scheduled_executor(new detail::thread_pool_attached_executor<
            policies::local_queue_scheduler<> >(first_thread, num_threads,
                priority, stacksize))
    {}
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_queue_attached_executor::static_queue_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority, thread_stacksize stacksize)
      : scheduled_executor(new detail::thread_pool_attached_executor<
            policies::static_queue_scheduler<> >(first_thread, num_threads,
                priority, stacksize))
    {}
#endif

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_attached_executor::local_priority_queue_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority, thread_stacksize stacksize)
      : scheduled_executor(new detail::thread_pool_attached_executor<
            policies::local_priority_queue_scheduler<> >(first_thread,
                num_threads, priority, stacksize))
    {}

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_attached_executor::static_priority_queue_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority, thread_stacksize stacksize)
      : scheduled_executor(new detail::thread_pool_attached_executor<
            policies::static_priority_queue_scheduler<> >(first_thread,
                num_threads, priority, stacksize))
    {}
#endif
}}}
