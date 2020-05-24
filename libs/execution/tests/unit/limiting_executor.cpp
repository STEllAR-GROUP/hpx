//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/thread_executors/limiting_executor.hpp>
#include <hpx/testing.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id thread_id_test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test2(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
}

void test_sync()
{
    auto exec1 = hpx::parallel::execution::parallel_executor(hpx::threads::thread_stacksize_small);
    auto exec2 = hpx::parallel::execution::default_executor(hpx::threads::thread_stacksize_large);

    hpx::threads::executors::limiting_executor<decltype(exec1)> lexec1(exec1, 100, 100);
    hpx::threads::executors::limiting_executor<decltype(exec2)> lexec2(exec2, 100, 100);

    hpx::async(lexec1, &test2, 42);
    hpx::async(lexec2, &test2, 42);

    auto id1 = hpx::parallel::execution::sync_execute(exec1, &thread_id_test, 42);
    auto id2 = hpx::this_thread::get_id();
    std::cout << "hpx::parallel::execution::sync_execute Ids " << id1 << " : " << id2 << std::endl;

    HPX_TEST(hpx::parallel::execution::sync_execute(exec1, &thread_id_test, 42) ==
        hpx::this_thread::get_id());
    HPX_TEST(hpx::parallel::execution::sync_execute(exec2, &thread_id_test, 42) ==
        hpx::this_thread::get_id());

    id1 = hpx::parallel::execution::sync_execute(lexec1, &thread_id_test, 42);
    id2 = hpx::this_thread::get_id();
    std::cout << "hpx::parallel::execution::sync_execute - limiting exec Ids " << id1 << " : " << id2 << std::endl;

    HPX_TEST(hpx::parallel::execution::sync_execute(lexec1, &thread_id_test, 42) ==
        hpx::this_thread::get_id());
    HPX_TEST(hpx::parallel::execution::sync_execute(lexec2, &thread_id_test, 42) ==
        hpx::this_thread::get_id());
}

void test_async()
{
    auto exec1 = hpx::parallel::execution::parallel_executor(hpx::threads::thread_stacksize_small);
    auto exec2 = hpx::parallel::execution::default_executor(hpx::threads::thread_stacksize_large);

    hpx::threads::executors::limiting_executor<decltype(exec1)> lexec1(exec1, 100, 100);
    hpx::threads::executors::limiting_executor<decltype(exec2)> lexec2(exec2, 100, 100);

    HPX_TEST(hpx::parallel::execution::async_execute(exec1, &thread_id_test, 42).get() !=
        hpx::this_thread::get_id());
    HPX_TEST(hpx::parallel::execution::async_execute(exec2, &thread_id_test, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test_f(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test_then()
{
    auto exec1 = hpx::parallel::execution::parallel_executor(hpx::threads::thread_stacksize_small);
    hpx::threads::executors::limiting_executor<decltype(exec1)> lexec1(exec1, 100, 100);

    hpx::future<void> f = hpx::make_ready_future();

    HPX_TEST(
        hpx::parallel::execution::then_execute(lexec1, &test_f, f, 42).get() ==
        hpx::this_thread::get_id());
}

void static_check_executor()
{
    using namespace hpx::traits;
    auto exec1 = hpx::parallel::execution::parallel_executor(hpx::threads::thread_stacksize_small);
    hpx::threads::executors::limiting_executor<decltype(exec1)> lexec1(exec1, 100, 100);

//    static_assert(has_sync_execute_member<executor>::value,
//        "has_sync_execute_member<executor>::value");
//    static_assert(has_async_execute_member<executor>::value,
//        "has_async_execute_member<executor>::value");
//    static_assert(!has_then_execute_member<executor>::value,
//        "!has_then_execute_member<executor>::value");
//    static_assert(has_bulk_sync_execute_member<executor>::value,
//        "has_bulk_sync_execute_member<executor>::value");
//    static_assert(has_bulk_async_execute_member<executor>::value,
//        "has_bulk_async_execute_member<executor>::value");
//    static_assert(!has_bulk_then_execute_member<executor>::value,
//        "!has_bulk_then_execute_member<executor>::value");
//    static_assert(has_post_member<executor>::value,
//        "check has_post_member<executor>::value");
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    static_check_executor();

    test_sync();
    test_async();
    test_then();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv, cfg), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
