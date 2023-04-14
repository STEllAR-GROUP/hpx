//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a HPX-thread printing
// "Hello World!" once. That's all.

//[hello_world_1_getting_started
// Including 'hpx/hpx_main.hpp' instead of the usual 'hpx/hpx_init.hpp' enables
// to use the plain C-main below as the direct main HPX entry point.
#include <hpx/hpx_main.hpp>
#include <hpx/future.hpp>
#include <hpx/execution/algorithms/as_sender.hpp>

#include <iostream>

namespace ex = hpx::execution::experimental;

std::uint64_t dummy_sender_fib(std::uint64_t n)
{
    if (n < 2)
        return n;

    auto res1 = hpx::this_thread::experimental::sync_wait(ex::just(dummy_sender_fib(n - 1))).value();
    auto res2 = hpx::this_thread::experimental::sync_wait(ex::just(dummy_sender_fib(n - 2))).value();

    return std::get<0>(res1) + std::get<0>(res2);
}

std::uint64_t future_fib(std::uint64_t n)
{
    if (n < 2)
        return n;

    auto f1 = hpx::async(/*hpx::launch::async,*/ future_fib, n - 1);
    auto f2 = hpx::async(future_fib, n - 2);

    return f1.get() + f2.get();
}

int main()
{
    // Say hello to the world!
    future_fib(20);
    //std::cout << dummy_sender_fib(20) << std::endl;
    std::cout << dummy_sender_fib(20) << std::endl;
    return 0;
}
//]
