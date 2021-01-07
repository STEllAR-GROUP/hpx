//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <thread>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return std::this_thread::get_id();
}

template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) !=
        std::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    HPX_TEST(hpx::parallel::execution::async_execute(exec, &test, 42).get() !=
        std::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
std::thread::id test_f(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return std::this_thread::get_id();
}

template <typename Executor>
void test_then(Executor& exec)
{
    hpx::future<void> f = hpx::make_ready_future();

    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get() !=
        std::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, std::thread::id tid, int passed_through)
{
    HPX_TEST_NEQ(tid, std::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    std::thread::id tid = std::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    std::thread::id tid = std::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, &bulk_test, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_then(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    hpx::parallel::execution::bulk_then_execute(
        exec, hpx::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42)
        .get();
    hpx::parallel::execution::bulk_then_execute(
        exec, &bulk_test_f, v, f, tid, 42)
        .get();
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_service_executor(Executor& exec)
{
    test_sync(exec);
    test_async(exec);
    test_then(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
    test_bulk_then(exec);
}

int hpx_main()
{
    using namespace hpx::parallel;
    using hpx::parallel::execution::service_executor_type;

#if defined(HPX_HAVE_IO_POOL)
    {
        execution::service_executor exec(service_executor_type::io_thread_pool);
        test_service_executor(exec);
    }
#endif

#if defined(HPX_HAVE_NETWORKING)
    if (hpx::is_networking_enabled())
    {
        execution::service_executor exec(
            service_executor_type::parcel_thread_pool);
        test_service_executor(exec);
    }
#endif

#if defined(HPX_HAVE_TIMER_POOL)
    {
        execution::service_executor exec(
            service_executor_type::timer_thread_pool);
        test_service_executor(exec);
    }
#endif

    {
        execution::service_executor exec(service_executor_type::main_thread);
        test_service_executor(exec);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
