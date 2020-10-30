//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_timed_sync(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::sync_execute_after(exec, milliseconds(1),
                 &test, 42) != hpx::this_thread::get_id());

    HPX_TEST(hpx::parallel::execution::sync_execute_at(exec,
                 steady_clock::now() + milliseconds(1), &test,
                 42) != hpx::this_thread::get_id());
}

template <typename Executor>
void test_timed_async(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::async_execute_after(
                 exec, milliseconds(1), &test, 42)
                 .get() != hpx::this_thread::get_id());
    HPX_TEST(hpx::parallel::execution::async_execute_at(
                 exec, steady_clock::now() + milliseconds(1), &test, 42)
                 .get() != hpx::this_thread::get_id());
}

template <typename Executor>
void test_timed_apply(Executor& exec)
{
    hpx::parallel::execution::post_after(exec, milliseconds(1), &test, 42);
    hpx::parallel::execution::post_at(
        exec, steady_clock::now() + milliseconds(1), &test, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_timed_thread_pool_executor(Executor& exec)
{
    test_timed_sync(exec);
    test_timed_async(exec);
    test_timed_apply(exec);
}

int hpx_main()
{
    std::size_t num_threads = hpx::get_os_thread_count();

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    {
        hpx::parallel::execution::static_queue_executor exec(num_threads);
        test_timed_thread_pool_executor(exec);
    }
#endif

    {
        hpx::parallel::execution::local_priority_queue_executor exec(
            num_threads);
        test_timed_thread_pool_executor(exec);
    }

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    {
        hpx::parallel::execution::static_priority_queue_executor exec(
            num_threads);
        test_timed_thread_pool_executor(exec);
    }
#endif

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
