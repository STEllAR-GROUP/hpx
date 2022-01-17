//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/semaphore.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> count(0);

void worker(hpx::lcos::local::counting_semaphore& sem)
{
    ++count;
    sem.signal();    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::lcos::local::counting_semaphore sem;

    for (std::size_t i = 0; i != 10; ++i)
        hpx::apply(&worker, std::ref(sem));

    // Wait for all threads to finish executing.
    sem.wait(10);

    HPX_TEST_EQ(count, 10);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
