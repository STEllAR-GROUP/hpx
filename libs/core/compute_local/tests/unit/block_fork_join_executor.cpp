//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/chrono.hpp>
#include <hpx/compute.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using hpx::execution::experimental::block_fork_join_executor;
using hpx::execution::experimental::fork_join_executor;

static std::atomic<std::size_t> count{0};

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, int passed_through)    //-V813
{
    ++count;
    HPX_TEST_EQ(passed_through, 42);
}

template <typename... ExecutorArgs>
void test_bulk_sync(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_sync\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::bind(&bulk_test, _1, _2), v, 42);
    HPX_TEST_EQ(count.load(), n);

    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, 42);
    HPX_TEST_EQ(count.load(), 2 * n);
}

template <typename... ExecutorArgs>
void test_bulk_async(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_async\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::bind(&bulk_test, _1, _2), v, 42))
        .get();
    HPX_TEST_EQ(count.load(), n);

    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec, &bulk_test, v, 42))
        .get();
    HPX_TEST_EQ(count.load(), 2 * n);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_exception(int, int passed_through)    //-V813
{
    HPX_TEST_EQ(passed_through, 42);
    throw std::runtime_error("test");
}

template <typename... ExecutorArgs>
void test_bulk_sync_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_sync_exception\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    bool caught_exception = false;
    try
    {
        hpx::parallel::execution::bulk_sync_execute(
            exec, &bulk_test_exception, v, 42);

        HPX_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename... ExecutorArgs>
void test_bulk_async_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_bulk_async_exception\n";

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};
    bool caught_exception = false;
    try
    {
        auto r = hpx::parallel::execution::bulk_async_execute(
            exec, &bulk_test_exception, v, 42);
        r.get();

        HPX_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

static std::atomic<std::size_t> count1{0};
static std::atomic<std::size_t> count2{0};
static std::atomic<std::size_t> count3{0};
static std::atomic<std::size_t> count4{0};

template <typename... ExecutorArgs>
void test_invoke_sync_homogeneous(ExecutorArgs&&... args)
{
    std::cerr << "test_invoke_sync_homogeneous\n";

    auto f1 = [] { ++count1; };

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};

    count1 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(1));

    count1 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(2));

    count1 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1, f1, f1, f1, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(5));

    count1 = 0;
    hpx::parallel::execution::sync_invoke(
        exec, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(11));
}

template <typename... ExecutorArgs>
void test_invoke_sync(ExecutorArgs&&... args)
{
    std::cerr << "test_invoke_sync\n";

    auto f1 = [] { ++count1; };
    auto f2 = [] { ++count2; };
    auto f3 = [] { ++count3; };
    auto f4 = [] { ++count4; };

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};

    count1 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(1));

    count1 = 0;
    count2 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1, f2);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(count2.load(), static_cast<std::size_t>(1));

    count1 = 0;
    count2 = 0;
    count3 = 0;
    count4 = 0;
    hpx::parallel::execution::sync_invoke(exec, f1, f2, f3, f4, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(count2.load(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(count3.load(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(count4.load(), static_cast<std::size_t>(1));

    count1 = 0;
    count2 = 0;
    count3 = 0;
    count4 = 0;
    hpx::parallel::execution::sync_invoke(
        exec, f1, f2, f3, f4, f1, f4, f1, f2, f3, f4, f1);
    HPX_TEST_EQ(count1.load(), static_cast<std::size_t>(4));
    HPX_TEST_EQ(count2.load(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(count3.load(), static_cast<std::size_t>(2));
    HPX_TEST_EQ(count4.load(), static_cast<std::size_t>(3));
}

template <typename... ExecutorArgs>
void test_invoke_sync_homogeneous_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_invoke_sync_homogeneous_exception\n";

    auto f1 = [] { throw std::runtime_error("test"); };

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};

    bool caught_exception = false;
    try
    {
        hpx::parallel::execution::sync_invoke(exec, f1, f1, f1);
        HPX_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename... ExecutorArgs>
void test_invoke_sync_exception(ExecutorArgs&&... args)
{
    std::cerr << "test_invoke_sync_exception\n";

    auto f1 = [] {};
    auto f2 = [] { throw std::runtime_error("test"); };

    block_fork_join_executor exec{std::forward<ExecutorArgs>(args)...};

    bool caught_exception = false;
    try
    {
        hpx::parallel::execution::sync_invoke(exec, f1, f2);
        HPX_TEST(false);
    }
    catch (std::runtime_error const& /*e*/)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename... ExecutorArgs>
void test_executor(hpx::threads::thread_priority priority,
    hpx::threads::thread_stacksize stacksize,
    fork_join_executor::loop_schedule schedule)
{
    std::cerr << "testing block_fork_join_executor with priority = " << priority
              << ", stacksize = " << stacksize << ", schedule = " << schedule
              << "\n";
    test_bulk_sync(priority, stacksize, schedule);
    test_bulk_async(priority, stacksize, schedule);
    test_bulk_sync_exception(priority, stacksize, schedule);
    test_bulk_async_exception(priority, stacksize, schedule);

    test_invoke_sync_homogeneous(priority, stacksize, schedule);
    test_invoke_sync(priority, stacksize, schedule);
    test_invoke_sync_homogeneous_exception(priority, stacksize, schedule);
    test_invoke_sync_exception(priority, stacksize, schedule);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // thread_stacksize::nostack cannot be used with the block_fork_join_executor
    // because it prevents other work from running when yielding. Using
    // thread_priority::low hangs for unknown reasons.
    for (auto const priority : {
             // hpx::threads::thread_priority::low,
             hpx::threads::thread_priority::bound,
             hpx::threads::thread_priority::normal,
             hpx::threads::thread_priority::high,
         })
    {
        for (auto const stacksize : {
                 // hpx::threads::thread_stacksize::nostack,
                 hpx::threads::thread_stacksize::small_,
             })
        {
            for (auto const schedule : {
                     fork_join_executor::loop_schedule::static_,
                     fork_join_executor::loop_schedule::dynamic,
                 })
            {
                test_executor(priority, stacksize, schedule);
            }
        }
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.force_min_os_threads!=4"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
