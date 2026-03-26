//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
void test_transform()
{
    // value
    auto f = hpx::make_ready_future(42);
    auto f2 = hpx::transform(
        std::move(f), [](int val) { return std::to_string(val); });
    HPX_TEST_EQ(f2.get(), "42");

    // void
    auto f_void = hpx::make_ready_future();
    auto f_void2 = hpx::transform(std::move(f_void), []() { return 10; });
    HPX_TEST_EQ(f_void2.get(), 10);
}

void test_and_then()
{
    // flatten: f returns a future, and_then auto-unwraps it
    auto f = hpx::make_ready_future(42);
    auto f2 = hpx::and_then(std::move(f),
        [](int val) { return hpx::make_ready_future(std::to_string(val)); });
    HPX_TEST_EQ(f2.get(), "42");
}

void test_or_else()
{
    // active exception with exception_ptr signature
    auto f = hpx::make_exceptional_future<int>(
        hpx::exception(hpx::error::bad_parameter, "testing or_else"));

    auto f2 = hpx::or_else(std::move(f), [](std::exception_ptr const& /*e*/) {
        return hpx::make_ready_future(100);
    });

    HPX_TEST_EQ(f2.get(), 100);

    // active exception with void argument signature
    auto f3 = hpx::make_exceptional_future<int>(
        hpx::exception(hpx::error::bad_parameter, "testing or_else"));

    auto f4 = hpx::or_else(
        std::move(f3), []() { return hpx::make_ready_future(200); });

    HPX_TEST_EQ(f4.get(), 200);

    // no exception (value passes through)
    auto f5 = hpx::make_ready_future(42);
    auto f6 = hpx::or_else(std::move(f5),
        [](std::exception_ptr const&) { return hpx::make_ready_future(1); });
    HPX_TEST_EQ(f6.get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
void test_shared_transform()
{
    hpx::shared_future<int> f = hpx::make_ready_future(42);
    auto f2 = hpx::transform(f, [](int val) { return std::to_string(val); });
    HPX_TEST_EQ(f2.get(), "42");
    HPX_TEST_EQ(f.get(), 42);
}

void test_shared_and_then()
{
    hpx::shared_future<int> f = hpx::make_ready_future(42);
    auto f2 = hpx::and_then(
        f, [](int val) { return hpx::make_ready_future(std::to_string(val)); });
    HPX_TEST_EQ(f2.get(), "42");
}

void test_shared_or_else()
{
    hpx::shared_future<int> f = hpx::make_exceptional_future<int>(
        hpx::exception(hpx::error::bad_parameter, "testing shared or_else"));

    auto f2 = hpx::or_else(f,
        [](std::exception_ptr const&) { return hpx::make_ready_future(300); });
    HPX_TEST_EQ(f2.get(), 300);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_transform();
    test_and_then();
    test_or_else();

    test_shared_transform();
    test_shared_and_then();
    test_shared_or_else();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
