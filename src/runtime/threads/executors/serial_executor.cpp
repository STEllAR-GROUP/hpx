//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/scheduling_loop.hpp>
#include <hpx/runtime/threads/executors/serial_executor.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    serial_executor::serial_executor(std::size_t num_queues)
      : state_(starting), scheduler_(num_queues)
    {
    }

    serial_executor::~serial_executor()
    {
        state_.store(stopping);
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void serial_executor::add(HPX_STD_FUNCTION<void()> f,
        char const* description)
    {
    }

    // Like add(), except that if the attempt to add the function would
    // cause the caller to block in add, try_add would instead do
    // nothing and return false.
    bool serial_executor::try_add(HPX_STD_FUNCTION<void()> f,
        char const* description)
    {
        return true;      // this function will never block
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void serial_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        HPX_STD_FUNCTION<void()> f, char const* description)
    {
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void serial_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        HPX_STD_FUNCTION<void()> f, char const* description)
    {
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t serial_executor::num_pending_tasks() const
    {
        return scheduler_.get_thread_count() -
            scheduler_.get_thread_count(terminated);
    }

    // execute all work
    void serial_executor::run(std::size_t num_thread)
    {
        boost::int64_t executed_threads = 0;
        boost::uint64_t overall_times = 0, thread_times = 0;
        scheduling_loop(num_thread, scheduler_, state_, 
            executed_threads, overall_times, thread_times);
    }
}}}}
