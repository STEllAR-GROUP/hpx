//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/functional/bind.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return passed_through;
}

using executor = hpx::parallel::execution::polymorphic_executor<int(int)>;

void test_sync(executor const& exec)
{
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) == 42);
}

void test_async(executor const& exec)
{
    HPX_TEST(
        hpx::parallel::execution::async_execute(exec, &test, 42).get() == 42);
}

///////////////////////////////////////////////////////////////////////////////
int test_f(hpx::shared_future<void> const& f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return 42;
}

void test_then(executor const& exec)
{
    hpx::future<void> f = hpx::make_ready_future();

    HPX_TEST(
        hpx::parallel::execution::then_execute(exec, &test_f, std::move(f), 42)
            .get() == 42);
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> count(0);

int bulk_test(std::size_t, int passed_through)
{
    ++count;
    HPX_TEST_EQ(passed_through, 42);
    return passed_through;
}

void test_bulk_sync(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(v.begin(), v.end(), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    count = 0;
    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::util::bind(&bulk_test, _1, _2), v, 42);
    HPX_TEST(count == v.size());

    count = 0;
    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, 42);
    HPX_TEST(count == v.size());
}

///////////////////////////////////////////////////////////////////////////////
void test_bulk_async(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    count = 0;
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::util::bind(&bulk_test, _1, _2), v, 42))
        .get();
    HPX_TEST(count == v.size());

    count = 0;
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec, &bulk_test, v, 42))
        .get();
    HPX_TEST(count == v.size());
}

///////////////////////////////////////////////////////////////////////////////
int bulk_test_f(
    std::size_t, hpx::shared_future<void> const& f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    ++count;
    HPX_TEST_EQ(passed_through, 42);
    return passed_through;
}

void test_bulk_then(executor const& exec)
{
    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    count = 0;
    hpx::parallel::execution::bulk_then_execute(
        exec, hpx::util::bind(&bulk_test_f, _1, _2, _3), v, f, 42)
        .get();
    HPX_TEST(count == v.size());

    count = 0;
    hpx::parallel::execution::bulk_then_execute(exec, &bulk_test_f, v, f, 42)
        .get();
    HPX_TEST(count == v.size());
}

void static_check_executor()
{
    using namespace hpx::traits;

    static_assert(has_sync_execute_member<executor>::value,
        "has_sync_execute_member<executor>::value");
    static_assert(has_async_execute_member<executor>::value,
        "has_async_execute_member<executor>::value");
    static_assert(has_then_execute_member<executor>::value,
        "has_then_execute_member<executor>::value");
    static_assert(has_bulk_sync_execute_member<executor>::value,
        "has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(has_bulk_then_execute_member<executor>::value,
        "has_bulk_then_execute_member<executor>::value");
    static_assert(has_post_member<executor>::value,
        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
void test_executor(executor const& exec)
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
    static_check_executor();

    hpx::execution::parallel_executor par_exec;
    test_executor(executor(par_exec));

    hpx::execution::sequenced_executor seq_exec;
    test_executor(executor(seq_exec));

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
