//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test illustrates #1112: Using thread_pool_executors causes segfault

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <iostream>

std::atomic<int> counter(0);

void doit()
{
    ++counter;
}

int main()
{
    {
        hpx::parallel::execution::local_priority_queue_executor exec(4);
        for (int i = 0; i != 100; ++i)
            hpx::async(exec, &doit);
    }

    HPX_TEST_EQ(counter.load(), 100);

    return hpx::util::report_errors();
}

