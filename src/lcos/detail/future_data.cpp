//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/launch_policy.hpp>

#include <utility>

namespace hpx { namespace lcos { namespace detail
{
    bool run_on_completed_on_new_thread(
        util::unique_function_nonser<bool()> && f, error_code& ec)
    {
        lcos::local::futures_factory<bool()> p(std::move(f));

        bool is_hpx_thread = nullptr != hpx::threads::get_self_ptr();
        hpx::launch policy = launch::fork;
        if (!is_hpx_thread)
            policy = launch::async;

        // launch a new thread executing the given function
        threads::thread_id_type tid = p.apply(
            policy, threads::thread_priority_boost,
            threads::thread_stacksize_current, ec);
        if (ec) return false;

        // wait for the task to run
        if (is_hpx_thread)
        {
            // make sure this thread is executed last
            hpx::this_thread::yield_to(thread::id(std::move(tid)));
            return p.get_future().get(ec);
        }
        else
        {
            // If we are not on a HPX thread, we need to return immediately, to
            // allow the newly spawned thread to execute. This might swallow
            // possible exceptions bubbling up from the completion handler (which
            // shouldn't happen anyway...
            return true;
        }
    }
}}}
