//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/executors/limiting_executor.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// this test launches many tasks continuously using a limiting executor
// some of the tasks will suspend themselves randomly and so new tasks will be
// spawned until the max tasks 'in flight' limit is reached. Once this happens
// no new tasks should be created and the number 'in flight' should remain
// below the limit. We track this using some counters and check that
// at the end of the test, the max counter never exceeded the limit.

// counters we will use to track tasks in flight
using atype = std::atomic<std::int64_t>;
static atype task_1_counter(0);
static atype task_1_total(0);
static atype task_1_max(0);
static const std::int64_t max1 = 110;

///////////////////////////////////////////////////////////////////////////////
//  simple task that can yield at random and increments counters
void test_fn(atype& acnt, atype& atot, atype& amax)
{
    // Increment active task count
    // set the max tasks running (not 100% atomic, but hopefully good enough)
    if (++acnt > amax)
        amax.store(acnt.load());
    ++atot;

    // yield some random amount of times to make the test more realistic
    // this allows other tasks to run and tests if the limiting exec
    // is doing its job
    std::default_random_engine eng;
    std::uniform_int_distribution<std::size_t> idist(10, 50);
    std::size_t loop = idist(eng);
    for (std::size_t i = 0; i < loop; ++i)
    {
        hpx::this_thread::yield();
    }

    // task is completing, decrement active task count
    --acnt;
}

void test_limit()
{
    auto exec1 = hpx::parallel::execution::parallel_executor(
        hpx::threads::thread_stacksize_small);
    //
    const bool block_on_exit = true;
    std::vector<hpx::future<void>> futures;
    // scope block for executor lifetime (block on destruction)
    {
        hpx::execution::experimental::limiting_executor<decltype(exec1)> lexec1(
            exec1, max1 / 2, max1, block_on_exit);

        // run this loop for N seconds and launch as many tasks as we can
        // then check that there were never more than N on a given executor
        bool ok = true;
        auto start = std::chrono::steady_clock::now();
        while (ok)
        {
            futures.push_back(
                hpx::async(lexec1, &test_fn, std::ref(task_1_counter),
                    std::ref(task_1_total), std::ref(task_1_max)));
            ok = (futures.size() < 50000) &&
                (std::chrono::steady_clock::now() - start <
                    std::chrono::milliseconds(500));
        }
        std::cout << "Reached end of launch with futures = " << futures.size()
                  << std::endl;
    }
    // the executors should block until all tasks have completed
    auto not_ready = std::count_if(
        futures.begin(), futures.end(), [](auto& f) { return !f.is_ready(); });
    // so all futures should be ready;
    HPX_TEST_EQ(not_ready, 0);

    // the max counters should not exceed the limit set
    // note that as the max set is not actually atomic, a race could allow the limit
    // to be exceeded by the number of threads in the worst case.
    std::cout << "Exec 1 had max " << task_1_max << " (allowed = " << max1
              << ") from a total of " << task_1_total << std::endl;
    HPX_TEST_LTE(task_1_max, max1);
    // if the test fails by a small number, then this would fix it
    // HPX_TEST_LTE(task_1_max, max1 + hpx::get_num_worker_threads());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_limit();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=cores"};

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv, cfg), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
