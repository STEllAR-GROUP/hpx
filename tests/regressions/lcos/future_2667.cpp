//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2667: Ambiguity of
// nested hpx::future<void>'s

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>
#include <utility>

#include <boost/atomic.hpp>

boost::atomic<bool> was_run(false);

void do_more_work()
{
    hpx::this_thread::sleep_for(std::chrono::seconds(1));
    was_run = true;
}

int main()
{
    hpx::future<hpx::future<void> > fut =
        hpx::async(
            []() -> hpx::future<void> {
                return hpx::async(
                    []() -> void {
                        do_more_work();
                    });
            });

    hpx::util::high_resolution_timer t;

    hpx::future<void> fut2 = std::move(fut);
    fut2.get();

    HPX_TEST(t.elapsed() > 1.0);
    HPX_TEST(was_run.load());

    return hpx::util::report_errors();
}
