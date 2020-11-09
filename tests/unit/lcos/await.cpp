// Copyright (C) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#if !defined(HPX_HAVE_AWAIT) && !defined(HPX_HAVE_CXX20_COROUTINES)
#error "This test requires compiler support for C++20 coroutines"
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos_local.hpp>
#include <hpx/include/threads.hpp>

#include <hpx/modules/testing.hpp>

#include <chrono>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int just_wait(int result)
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return result;
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<int> test1()
{
    co_return co_await hpx::make_ready_future(42);
}

hpx::future<int> test2()
{
    auto result = co_await hpx::make_ready_future(42);
    (void) result;
    co_return result;
}

hpx::future<int> test3()
{
    int local_variable[128] = {42};      // large local variable
    auto result = co_await hpx::make_ready_future(local_variable[0]);
    (void) result;
    co_return result;
}

hpx::future<int> async_test1()
{
    co_return co_await hpx::async(just_wait, 42);
}

hpx::future<int> async_test2()
{
    auto result = co_await hpx::async(just_wait, 42);
    (void) result;
    co_return result;
}

hpx::future<int> async_test3()
{
    int local_variable[128] = {42};      // large local variable
    auto result = co_await hpx::async(just_wait, local_variable[0]);
    (void) result;
    co_return result;
}

void simple_await_tests()
{
    HPX_TEST_EQ(test1().get(), 42);
    HPX_TEST_EQ(test2().get(), 42);
    HPX_TEST_EQ(test3().get(), 42);

    HPX_TEST_EQ(async_test1().get(), 42);
    HPX_TEST_EQ(async_test2().get(), 42);
    HPX_TEST_EQ(async_test3().get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<int> fib1(int n)
{
    if (n >= 2)
        n = co_await fib1(n - 1) + co_await fib1(n - 2);
    (void) n;
    co_return n;
}

hpx::future<int> fib2(int n)
{
    if (n >= 2)
        n = co_await hpx::async(&fib2, n - 1) + co_await fib2(n - 2);
    (void) n;
    co_return n;
}

void simple_recursive_await_tests()
{
    HPX_TEST_EQ(fib1(10).get(), 55);
    HPX_TEST_EQ(fib2(10).get(), 55);
}

///////////////////////////////////////////////////////////////////////////////
hpx::shared_future<int> shared_test1()
{
    co_return co_await hpx::make_ready_future(42);
}

hpx::shared_future<int> shared_test2()
{
    auto result = co_await hpx::make_ready_future(42);
    (void) result;
    co_return result;
}

hpx::shared_future<int> shared_test3()
{
    int local_variable[128] = {42};      // large local variable
    auto result = co_await hpx::make_ready_future(local_variable[0]);
    (void) result;
    co_return result;
}

hpx::shared_future<int> async_shared_test1()
{
    co_return co_await hpx::async(just_wait, 42);
}

hpx::shared_future<int> async_shared_test2()
{
    auto result = co_await hpx::async(just_wait, 42);
    (void) result;
    co_return result;
}

hpx::shared_future<int> async_shared_test3()
{
    int local_variable[128] = {42};      // large local variable
    auto result = co_await hpx::async(just_wait, local_variable[0]);
    (void) result;
    co_return result;
}

void simple_await_shared_tests()
{
    HPX_TEST_EQ(shared_test1().get(), 42);
    HPX_TEST_EQ(shared_test2().get(), 42);
    HPX_TEST_EQ(shared_test3().get(), 42);

    HPX_TEST_EQ(async_shared_test1().get(), 42);
    HPX_TEST_EQ(async_shared_test2().get(), 42);
    HPX_TEST_EQ(async_shared_test3().get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
hpx::shared_future<int> shared_fib1(int n)
{
    if (n >= 2)
        n = co_await shared_fib1(n - 1) + co_await shared_fib1(n - 2);
    (void) n;
    co_return n;
}

hpx::shared_future<int> shared_fib2(int n)
{
    if (n >= 2)
        n = co_await hpx::async(&fib2, n - 1) + co_await shared_fib2(n - 2);
    (void) n;
    co_return n;
}

void simple_recursive_await_shared_tests()
{
    HPX_TEST_EQ(shared_fib1(10).get(), 55);
    HPX_TEST_EQ(shared_fib2(10).get(), 55);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    simple_await_tests();
    simple_recursive_await_tests();

    simple_await_shared_tests();
    simple_recursive_await_shared_tests();

    HPX_TEST_EQ(hpx::finalize(), 0);
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
