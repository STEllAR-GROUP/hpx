//  Copyright (c) 2019 Christopher Hinz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #3634:
// The build fails if shared_future<>::then is called with a thread executor

#include <hpx/hpx_main.hpp>

#include <hpx/include/thread_executors.hpp>
#include <hpx/include/lcos.hpp>

int main()
{
    hpx::parallel::execution::pool_executor executor("default");

    auto future = hpx::make_ready_future().share();

    future.then(executor, [](hpx::shared_future<void> future) { future.get(); });

    return 0;
}
