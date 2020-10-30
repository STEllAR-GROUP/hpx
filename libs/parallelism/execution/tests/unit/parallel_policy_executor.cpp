//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Policy>
void test_sync()
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    executor exec;
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) ==
        hpx::this_thread::get_id());
}

template <typename Policy>
void test_async(bool sync)
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    executor exec;
    bool result =
        hpx::parallel::execution::async_execute(exec, &test, 42).get() ==
        hpx::this_thread::get_id();

    HPX_TEST_EQ(sync, result);
}

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test_f(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Policy>
void test_then(bool sync)
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    hpx::future<void> f = hpx::make_ready_future();

    executor exec;
    bool result =
        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get() ==
        hpx::this_thread::get_id();

    HPX_TEST_EQ(sync, result);
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_s(int, hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void bulk_test_a(int, hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Policy>
void test_bulk_sync(bool sync)
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::parallel::execution::bulk_sync_execute(exec,
        hpx::util::bind(sync ? &bulk_test_s : &bulk_test_a, _1, tid, _2), v,
        42);
    hpx::parallel::execution::bulk_sync_execute(
        exec, sync ? &bulk_test_s : &bulk_test_a, v, tid, 42);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
void test_bulk_async(bool sync)
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec,
            hpx::util::bind(sync ? &bulk_test_s : &bulk_test_a, _1, tid, _2), v,
            42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, sync ? &bulk_test_s : &bulk_test_a, v, tid, 42))
        .get();
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f_s(int, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void bulk_test_f_a(int, hpx::shared_future<void> f, hpx::thread::id tid,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Policy>
void test_bulk_then(bool sync)
{
    typedef hpx::execution::parallel_policy_executor<Policy> executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    executor exec;
    hpx::parallel::execution::bulk_then_execute(exec,
        hpx::util::bind(
            sync ? &bulk_test_f_s : &bulk_test_f_a, _1, _2, tid, _3),
        v, f, 42)
        .get();
    hpx::parallel::execution::bulk_then_execute(
        exec, sync ? &bulk_test_f_s : &bulk_test_f_a, v, f, tid, 42)
        .get();
}

template <typename Policy>
void static_check_executor()
{
    using namespace hpx::traits;
    using executor = hpx::execution::parallel_policy_executor<Policy>;

    static_assert(has_sync_execute_member<executor>::value,
        "has_sync_execute_member<executor>::value");
    static_assert(has_async_execute_member<executor>::value,
        "has_async_execute_member<executor>::value");
    static_assert(has_then_execute_member<executor>::value,
        "has_then_execute_member<executor>::value");
    static_assert(!has_bulk_sync_execute_member<executor>::value,
        "!has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(has_bulk_then_execute_member<executor>::value,
        "has_bulk_then_execute_member<executor>::value");
    static_assert(has_post_member<executor>::value,
        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
void policy_test(bool sync = false)
{
    static_check_executor<Policy>();

    test_sync<Policy>();
    test_async<Policy>(sync);
    test_then<Policy>(sync);

    test_bulk_sync<Policy>(sync);
    test_bulk_async<Policy>(sync);
    test_bulk_then<Policy>(sync);
}

int hpx_main()
{
    policy_test<hpx::launch>();

    policy_test<hpx::launch::async_policy>();
    policy_test<hpx::launch::sync_policy>(true);
    policy_test<hpx::launch::fork_policy>();
    policy_test<hpx::launch::deferred_policy>(true);

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
