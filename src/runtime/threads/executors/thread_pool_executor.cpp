//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executor.hpp>
#include <hpx/util/register_locks.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    void get_processing_units(std::size_t num_threads,
        std::vector<std::size_t>& punits)
    {
        for (std::size_t i = 0; i < num_threads; ++i)
            punits.push_back(i);
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_pool_executor::thread_pool_executor(std::size_t num_threads)
      : scheduler_(num_threads), shutdown_sem_(0), state_(running)
    {
        std::vector<std::size_t> processing_units;
        get_processing_units(num_threads, processing_units);

        for (std::size_t i = 0; i < num_threads; ++i) {
            register_thread_nullary(util::bind(&thread_pool_executor::run, this, i),
                "thread_pool_executor thread", threads::pending, true,
                threads::thread_priority_normal, processing_units[i]);
        }
    }

    thread_pool_executor::~thread_pool_executor()
    {
        state_.store(stopping);
        shutdown_sem_.wait();
    }

    static inline threads::thread_state_enum thread_function_nullary(
        HPX_STD_FUNCTION<void()> const& func)
    {
        // execute the actual thread function
        func();

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::terminated;
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void thread_pool_executor::add(HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        thread_init_data data(util::bind(
            &thread_function_nullary, boost::move(f)), desc);

        threads::detail::create_thread(scheduler_, data);
    }

    // Like add(), except that if the attempt to add the function would
    // cause the caller to block in add, try_add would instead do
    // nothing and return false.
    bool thread_pool_executor::try_add(HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        thread_init_data data(util::bind(
            &thread_function_nullary, boost::move(f)), desc);

        threads::detail::create_thread(scheduler_, data);
        return true;      // this function will never block
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void thread_pool_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        thread_init_data data(util::bind(
            &thread_function_nullary, boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended);
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, abs_time, id);
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void thread_pool_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        thread_init_data data(util::bind(
            &thread_function_nullary, boost::move(f)), desc);

        thread_id_type id = threads::detail::create_thread(
            scheduler_, data, suspended);
        BOOST_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, rel_time, id);
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t thread_pool_executor::num_pending_tasks() const
    {
        return scheduler_.get_thread_count() -
            scheduler_.get_thread_count(terminated);
    }

    // execute all work
    void thread_pool_executor::run(std::size_t num_thread)
    {
        boost::int64_t executed_threads = 0;
        boost::uint64_t overall_times = 0, thread_times = 0;
        threads::detail::scheduling_loop(num_thread, scheduler_, state_,
            executed_threads, overall_times, thread_times);
        shutdown_sem_.signal();
    }
}}}}
