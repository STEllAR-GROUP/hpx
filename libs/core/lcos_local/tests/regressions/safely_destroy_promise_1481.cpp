//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1481:
// Sync primitives safe destruction

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>

void test_safe_destruction()
{
    hpx::thread t;
    hpx::future<void> outer;

    {
        hpx::promise<void> p;
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

int hpx_main()
{
    test_safe_destruction();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
