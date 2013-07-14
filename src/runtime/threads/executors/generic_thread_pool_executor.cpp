//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/executors/generic_thread_pool_executor.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/register_locks.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    generic_thread_pool_executor::generic_thread_pool_executor(
            policies::scheduler_base* scheduler)
      : scheduler_base_(scheduler)
    {}

    generic_thread_pool_executor::~generic_thread_pool_executor()
    {}

    threads::thread_state_enum
    generic_thread_pool_executor::thread_function_nullary(
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
    void generic_thread_pool_executor::add(
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, error_code& ec)
    {
        // create a new thread
        thread_init_data data(util::bind(
            &generic_thread_pool_executor::thread_function_nullary,
            boost::move(f)), desc);

        threads::detail::create_thread(scheduler_base_, data,
            initial_state, run_now, ec);
        if (ec) return;

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t generic_thread_pool_executor::num_pending_closures(error_code& ec) const
    {
        return scheduler_base_->get_thread_count() -
                    scheduler_base_->get_thread_count(terminated);
    }
}}}}

namespace hpx { namespace threads { namespace executors
{
    ///////////////////////////////////////////////////////////////////////////
    // this is just a wrapper around a scheduler_base assuming the wrapped
    // scheduler outlives the wrapper
    generic_thread_pool_executor::generic_thread_pool_executor(
            policies::scheduler_base* scheduler)
      : executor(new detail::generic_thread_pool_executor(scheduler))
    {}
}}}
