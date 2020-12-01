//  Copyright (c)      2020 ETH Zurich
//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

static std::atomic<std::size_t> count{0};

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, int passed_through)    //-V813
{
    ++count;
    HPX_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    using executor = hpx::execution::experimental::fork_join_executor;

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::util::bind(&bulk_test, _1, _2), v, 42);
    HPX_TEST_EQ(count.load(), n);

    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, 42);
    HPX_TEST_EQ(count.load(), 2 * n);
}

void test_bulk_async()
{
    using executor = hpx::execution::experimental::fork_join_executor;

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      exec, hpx::util::bind(&bulk_test, _1, _2), v, 42))
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

void test_bulk_sync_exception()
{
    using executor = hpx::execution::experimental::fork_join_executor;

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    executor exec;
    bool caught_exception = false;
    try
    {
        hpx::parallel::execution::bulk_sync_execute(
            exec, &bulk_test_exception, v, 42);

        HPX_TEST(false);
    }
    catch (std::runtime_error const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

void test_bulk_async_exception()
{
    using executor = hpx::execution::experimental::fork_join_executor;

    count = 0;
    std::size_t const n = 107;
    std::vector<int> v(n);
    std::iota(std::begin(v), std::end(v), std::rand());

    executor exec;
    bool caught_exception = false;
    try
    {
        auto r = hpx::parallel::execution::bulk_async_execute(
            exec, &bulk_test_exception, v, 42);
        HPX_TEST_EQ(r.size(), std::size_t(1));
        r[0].get();

        HPX_TEST(false);
    }
    catch (std::runtime_error const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

void static_check_executor()
{
    using namespace hpx::traits;
    using executor = hpx::execution::experimental::fork_join_executor;

    static_assert(!has_sync_execute_member<executor>::value,
        "!has_sync_execute_member<executor>::value");
    static_assert(!has_async_execute_member<executor>::value,
        "!has_async_execute_member<executor>::value");
    static_assert(!has_then_execute_member<executor>::value,
        "!has_then_execute_member<executor>::value");
    static_assert(has_bulk_sync_execute_member<executor>::value,
        "has_bulk_sync_execute_member<executor>::value");
    static_assert(has_bulk_async_execute_member<executor>::value,
        "has_bulk_async_execute_member<executor>::value");
    static_assert(!has_bulk_then_execute_member<executor>::value,
        "!has_bulk_then_execute_member<executor>::value");
    static_assert(
        !has_post_member<executor>::value, "!has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    static_check_executor();

    test_bulk_sync();
    test_bulk_async();
    test_bulk_sync_exception();
    test_bulk_async_exception();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
