//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2667: Ambiguity of
// nested hpx::future<void>'s

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <chrono>
#include <utility>

std::atomic<bool> was_run(false);

void do_more_work()
{
    hpx::this_thread::sleep_for(std::chrono::seconds(1));
    was_run = true;
}

int hpx_main()
{
    hpx::future<hpx::future<void>> fut = hpx::async([]() -> hpx::future<void> {
        return hpx::async([]() -> void { do_more_work(); });
    });

    hpx::chrono::high_resolution_timer t;

    hpx::future<void> fut2 = std::move(fut);
    fut2.get();

    HPX_TEST_LT(1.0, t.elapsed());
    HPX_TEST(was_run.load());

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
