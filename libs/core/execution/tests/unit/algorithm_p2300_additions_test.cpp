//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_STDEXEC)

#include <hpx/execution/algorithms/continues_on.hpp>
#include <hpx/execution/algorithms/into_variant.hpp>
#include <hpx/execution/algorithms/just.hpp>
#include <hpx/execution/algorithms/read_env.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution/algorithms/starts_on.hpp>
#include <hpx/execution/algorithms/stopped_as_error.hpp>
#include <hpx/execution/algorithms/stopped_as_optional.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/algorithms/upon_error.hpp>
#include <hpx/execution/algorithms/upon_stopped.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace ex = hpx::execution::experimental;

void test_upon_error()
{
    ex::run_loop loop;
    auto sched = loop.get_scheduler();

    {
        auto result = ex::sync_wait(
            ex::upon_error(ex::just(42), [](std::exception_ptr) { return 0; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        auto result = ex::sync_wait(
            ex::upon_error(ex::just_error(std::runtime_error("test")),
                [](std::exception_ptr) { return 99; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 99);
    }
}

void test_upon_stopped()
{
    {
        auto result =
            ex::sync_wait(ex::upon_stopped(ex::just(42), []() { return -1; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }

    {
        auto result = ex::sync_wait(
            ex::upon_stopped(ex::just_stopped(), []() { return 100; }));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 100);
    }
}

void test_stopped_as_optional()
{
    {
        auto result = ex::sync_wait(ex::stopped_as_optional(ex::just(42)));
        HPX_TEST(result.has_value());
        auto opt = hpx::get<0>(*result);
        HPX_TEST(opt.has_value());
        HPX_TEST_EQ(*opt, 42);
    }

    {
        auto result =
            ex::sync_wait(ex::stopped_as_optional(ex::just_stopped()));
        HPX_TEST(result.has_value());
        auto opt = hpx::get<0>(*result);
        HPX_TEST(!opt.has_value());
    }
}

void test_stopped_as_error()
{
    {
        auto result = ex::sync_wait(
            ex::stopped_as_error(ex::just(42), std::runtime_error("stopped")));
        HPX_TEST(result.has_value());
        HPX_TEST_EQ(hpx::get<0>(*result), 42);
    }
}

void test_continues_on()
{
    ex::run_loop loop;
    auto sched = loop.get_scheduler();

    bool executed = false;
    ex::start_detached(ex::continues_on(
        ex::just() | ex::then([&] { executed = true; }), sched));

    loop.run();
    HPX_TEST(executed);
}

void test_starts_on()
{
    ex::run_loop loop;
    auto sched = loop.get_scheduler();

    bool executed = false;
    ex::start_detached(
        ex::starts_on(sched, ex::just() | ex::then([&] { executed = true; })));

    loop.run();
    HPX_TEST(executed);
}

void test_into_variant()
{
    {
        auto result = ex::sync_wait(ex::into_variant(ex::just(42)));
        HPX_TEST(result.has_value());
    }

    {
        auto result = ex::sync_wait(ex::into_variant(ex::just(42, 3.14)));
        HPX_TEST(result.has_value());
    }
}

void test_read_env()
{
    auto sender = ex::read_env([](auto const&) { return 42; });
    auto result = ex::sync_wait(sender);
    HPX_TEST(result.has_value());
    HPX_TEST_EQ(hpx::get<0>(*result), 42);
}

int main()
{
    test_upon_error();
    test_upon_stopped();
    test_stopped_as_optional();
    test_stopped_as_error();
    test_continues_on();
    test_starts_on();
    test_into_variant();
    test_read_env();

    return hpx::util::report_errors();
}

#else

int main()
{
    return 0;
}

#endif
