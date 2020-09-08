//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1481:
// Sync primitives safe destruction

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos_local.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>

void test_safe_destruction()
{
    hpx::thread t;
    hpx::future<void> outer;

    {
        hpx::lcos::local::promise<void> p;
        hpx::shared_future<void> inner = p.get_future().share();

        // Delay returning from p.set_value() below to destroy the promise
        // before set_value returns.
        outer = inner.then([](hpx::shared_future<void>&&) {
            hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
        });

        // create a thread which will make the inner future ready
        t = hpx::thread([&p]() { p.set_value(); });
        inner.get();
    }

    outer.get();
    t.join();
}

int main()
{
    test_safe_destruction();
    return hpx::util::report_errors();
}
