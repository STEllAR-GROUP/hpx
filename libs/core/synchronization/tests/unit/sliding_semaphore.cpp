//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/async_local/apply.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> count(0);

void worker(hpx::lcos::local::sliding_semaphore& sem)
{
    sem.signal(++count);    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::lcos::local::sliding_semaphore sem(9);

    for (std::size_t i = 0; i != 10; ++i)
        hpx::apply(&worker, std::ref(sem));

    // Wait for all threads to finish executing.
    sem.wait(19);

    HPX_TEST_EQ(count, 10);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
