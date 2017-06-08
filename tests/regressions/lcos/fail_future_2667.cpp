//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2667: Ambiguity of
// nested hpx::future<void>'s.
//
// This test is supposed to fail compiling.

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <chrono>
#include <utility>

int main()
{
    hpx::future<hpx::future<int> > fut =
        hpx::async(
            []() -> hpx::future<int> {
                return hpx::async(
                    []() -> int {
                        return 42;
                    });
            });

    hpx::future<void> fut2 = std::move(fut);
    fut2.get();

    return 0;
}
