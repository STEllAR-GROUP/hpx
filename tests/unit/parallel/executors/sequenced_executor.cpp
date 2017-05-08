//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test_sync()
{
    hpx::parallel::execution::sequenced_executor exec;
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) ==
        hpx::this_thread::get_id());
}

void test_async()
{
    typedef hpx::parallel::execution::parallel_executor executor;

    executor exec(hpx::launch::fork);
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

void test_then()
{
    typedef hpx::parallel::execution::sequenced_executor executor;

    hpx::future<void> f = hpx::make_ready_future();

    executor exec;
    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get() ==
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid == hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::parallel::execution::sequenced_executor exec;
    hpx::parallel::execution::sync_bulk_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::sync_bulk_execute(
        exec, &bulk_test, v, tid, 42);
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

void test_bulk_then()
{
    typedef hpx::parallel::execution::sequenced_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    executor exec;
    hpx::parallel::execution::then_bulk_execute(
        exec, hpx::util::bind(&bulk_test_f, _1, _2, tid, _3), v, f, 42
    ).get();
    hpx::parallel::execution::then_bulk_execute(
        exec, &bulk_test_f, v, f, tid, 42
    ).get();
}

void test_bulk_async()
{
    typedef hpx::parallel::execution::sequenced_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        hpx::parallel::execution::async_bulk_execute(
            exec, &bulk_test, v, tid, 42)
    ).get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_sync();
    test_async();
    test_then();

    test_bulk_sync();
    test_bulk_async();
    test_bulk_then();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
