//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<int> count(0);
int const initial_count = 42;
int const num_tasks = 139;
std::atomic<int> completed_tasks(0);

void worker(hpx::sliding_semaphore& sem)
{
    sem.signal(++count);    // signal main thread
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::vector<hpx::future<void>> futures;
    futures.reserve(num_tasks);

    hpx::sliding_semaphore sem(initial_count);

    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        futures.emplace_back(hpx::async(&worker, std::ref(sem)));
    }

    sem.wait(initial_count + num_tasks);

    HPX_TEST_EQ(count, num_tasks);

    // Since sem.signal(++count) (in worker) is not an atomic operation we wait
    // for the tasks to finish here. The task which signals the count that
    // releases the waiting thread is not necessarily the last one to signal the
    // semaphore. The following can happen:
    //
    //   thread 0             thread 1                thread 2
    //   -------------------  ----------------------- ---------------------
    //   atomic<int> count(0)
    //   semaphore sem(0)
    //   sem.wait(2)
    //        .               new_count = ++count
    //        .                                       new_count = ++count
    //        .                                       sem.signal(new_count)
    //   sem.wait(2) returns
    //   sem destructed
    //                        (sem is a dangling ref)
    //                        sem.signal(new_count)
    //
    hpx::wait_all(std::move(futures));

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
