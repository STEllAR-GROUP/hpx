//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/detail/future_data.hpp>

namespace hpx { namespace lcos { namespace detail
{
    bool run_on_completed_on_new_thread(
        util::unique_function_nonser<bool()> && f, error_code& ec)
    {
        lcos::local::futures_factory<bool()> p(std::move(f));

        // launch a new thread executing the given function
        p.apply(launch::fork, threads::thread_priority_boost,
            threads::thread_stacksize_default, ec);
        if (ec) return false;

        // wait for the task to run
        return p.get_future().get(ec);
    }
}}}
