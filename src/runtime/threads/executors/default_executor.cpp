//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/executors/default_executor.hpp>

#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    default_executor::default_executor()
      : scheduled_executor_base()
    {}

    default_executor::default_executor(thread_priority priority,
        thread_stacksize stacksize, thread_schedule_hint schedulehint)
      : scheduled_executor_base(priority, stacksize, schedulehint)
    {}

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void default_executor::add(closure_type&& f,
        util::thread_description const& desc,
        threads::thread_state_enum initial_state, bool run_now,
        threads::thread_stacksize stacksize,
        threads::thread_schedule_hint schedulehint,
        error_code& ec)
    {
        if (stacksize == threads::thread_stacksize_default)
            stacksize = stacksize_;

        register_thread_nullary(std::move(f), desc, initial_state, run_now,
            priority_, schedulehint, stacksize, ec);
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void default_executor::add_at(
        util::steady_clock::time_point const& abs_time,
        closure_type&& f, util::thread_description const& description,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        if (stacksize == threads::thread_stacksize_default)
            stacksize = stacksize_;

        // create new thread
        thread_id_type id = register_thread_nullary(
            std::move(f), description, suspended, false,
            priority_, schedulehint_, stacksize, ec);
        if (ec) return;

        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // now schedule new thread for execution
        set_thread_state(id, abs_time);
    }

    // Return an estimate of the number of waiting tasks.
    std::uint64_t default_executor::num_pending_closures(error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return get_thread_count() - get_thread_count(terminated);
    }

    // Reset internal (round robin) thread distribution scheme
    void default_executor::reset_thread_distribution()
    {
        threads::reset_thread_distribution();
    }

    // Set the new scheduler mode
    void default_executor::set_scheduler_mode(
        threads::policies::scheduler_mode mode)
    {
        threads::set_scheduler_mode(mode);
    }


    // Return the requested policy element
    std::size_t default_executor::get_policy_element(
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
            "default_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }
}}}}
