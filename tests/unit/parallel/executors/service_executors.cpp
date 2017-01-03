//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/executors/service_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <boost/range/functions.hpp>
#include <boost/thread/thread.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_sync(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::sync_execute(exec, &test, 42) !=
        hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    HPX_TEST(
        hpx::parallel::execution::async_execute(exec, &test, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test_f(hpx::future<void> f, int passed_through)
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_then(Executor& exec)
{
    hpx::future<void> f = hpx::make_ready_future();

    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::sync_bulk_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::sync_bulk_execute(
        exec, &bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::when_all(hpx::parallel::execution::async_bulk_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(hpx::parallel::execution::async_bulk_execute(
        exec, &bulk_test, v, tid, 42)
    ).get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int value, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through) //-V813
{
    HPX_ASSERT(f.is_ready());   // make sure, future is ready

    f.get();                    // propagate exceptions

    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_then(Executor& exec)
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    hpx::parallel::execution::then_bulk_execute(
        exec, hpx::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42
    ).get();
    hpx::parallel::execution::then_bulk_execute(
        exec, &bulk_test_f, v, f, tid, 42
    ).get();
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

int hpx_main(int argc, char* argv[])
{
    using namespace hpx::parallel;
    using hpx::threads::executors::service_executor_type;

    {
        execution::service_executor exec(service_executor_type::io_thread_pool);
        test_service_executor(exec);
    }

    {
        execution::service_executor exec(service_executor_type::parcel_thread_pool);
        test_service_executor(exec);
    }

    {
        execution::service_executor exec(service_executor_type::timer_thread_pool);
        test_service_executor(exec);
    }

    {
        execution::service_executor exec(service_executor_type::main_thread);
        test_service_executor(exec);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
