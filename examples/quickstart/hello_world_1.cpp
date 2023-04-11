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

std::uint64_t sender_fib(std::uint64_t n)
{
    if (n < 2)
        return n;

    //std::vector<hpx::future<std::uint64_t>> senders;

    std::vector<ex::any_sender<std::uint64_t>> senders;

    senders.emplace_back(ex::just(sender_fib(n - 1)));
    senders.emplace_back(ex::just(sender_fib(n - 2)));

    //senders.emplace_back(ex::as_sender(hpx::async(sender_fib, n - 2)));

    auto cont = ex::when_all_vector(std::move(senders));

    //auto cont2 = hpx::execution::experimental::then(cont, [](auto... args) {return args; });
    //auto s1 = hpx::execution::experimental::as_sender(hpx::async(sender_fib, n - 2));

     ex::start_detached(cont);
    // return std::get<0>(res);
    return 42;
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
    std::cout << future_fib(20) << std::endl;
    //std::cout << sender_fib(20) << std::endl;
    sender_fib(20);
    return 0;
}
//]
