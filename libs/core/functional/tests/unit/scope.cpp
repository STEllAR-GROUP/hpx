//  Copyright (c) 2024-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Copyright (c) 2020 Martin Moene
//  This is inspired by https://github.com/martinmoene/scope-lite

#include <hpx/config.hpp>
#include <hpx/experimental/scope.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>

static bool is_called = false;

namespace on {

    void exit()
    {
        is_called = true;
    }

    void fail()
    {
        is_called = true;
    }

    void success()
    {
        is_called = true;
    }
}    // namespace on

#if __cplusplus >= 202302L
namespace cexpr {

    bool is_called_exit()
    {
        bool result = false;
        {
            auto change = hpx::experimental::scope_exit([&] { result = true; });
        }
        return result;
    }

    bool is_not_called_fail()
    {
        bool result = false;
        {
            auto change = hpx::experimental::scope_fail([&] { result = true; });
        }
        return result;
    }

    bool is_called_success()
    {
        bool result = false;
        {
            auto guard =
                hpx::experimental::scope_success([&]() { result = true; });
        }
        return result;
    }
}    // namespace cexpr
#endif

// scope_exit: exit function is called at end of scope
void scope_exit_called()
{
    is_called = false;

    // scope:
    {
        auto guard = hpx::experimental::scope_exit(on::exit);
    }

    HPX_TEST(is_called);
}

// scope_exit: exit function is called at end of scope (lambda)
void scope_exit_called_lambda()
{
    is_called = false;

    {
        auto guard = hpx::experimental::scope_exit([]() { is_called = true; });
    }

    HPX_TEST(is_called);
}

// scope_exit: exit function is called when an exception occurs
void scope_exit_called_exception()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_exit(on::exit);
        throw std::exception();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

// scope_exit: exit function is not called at end of scope when released
void scope_exit_not_called_released()
{
    is_called = false;

    {
        auto guard = hpx::experimental::scope_exit(on::exit);
        guard.release();
    }

    HPX_TEST(!is_called);
}

// scope_fail: exit function is called when an exception occurs
void scope_fail_called_exception()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_fail(on::fail);
        throw std::exception();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

// scope_fail: exit function is called when an exception occurs (lambda)
void scope_fail_called_exception_lambda()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_fail([]() { is_called = true; });
        throw std::exception();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

// scope_fail: exit function is not called when no exception occurs
void scope_fail_not_called_no_exception()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_fail(on::fail);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(!is_called);
}

// scope_fail: exit function is not called when released
void scope_fail_not_called_released()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_fail(on::fail);
        guard.release();

        throw std::exception();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(!is_called);
}

// scope_success: exit function is called when no exception occurs
void scope_success_called_no_exception()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_success(on::success);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

// scope_success: exit function is called when no exception occurs (lambda)
void scope_success_called_no_exception_lambda()
{
    is_called = false;

    try
    {
        auto guard =
            hpx::experimental::scope_success([]() { is_called = true; });
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

// scope_success: exit function is not called when an exception occurs
void scope_success_not_called_exception()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_success(on::success);
        throw std::exception();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(!is_called);
}

// scope_success: exit function is not called when released
void scope_success_not_called_released()
{
    is_called = false;

    try
    {
        auto guard = hpx::experimental::scope_success(on::success);
        guard.release();
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(!is_called);
}

// scope_success: exit function can throw (lambda)
void scope_success_exit_throws()
{
    is_called = false;

    try
    {
        // skipped_guard is expected to not be called, as the destructor of
        // guard will throw.
        auto skipped_guard =
            hpx::experimental::scope_success([]() { is_called = false; });

        auto guard = hpx::experimental::scope_success([]() {
            is_called = true;
            throw std::exception();
        });
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (...)
    {
    }

    HPX_TEST(is_called);
}

int main()
{
#if __cplusplus >= 202302L
    HPX_TEST(cexpr::is_called_exit());
    HPX_TEST(!cexpr::is_not_called_fail());
    HPX_TEST(cexpr::is_called_success());
#endif

    scope_exit_called();
    scope_exit_called_lambda();
    scope_exit_called_exception();
    scope_exit_not_called_released();

    scope_fail_called_exception();
    scope_fail_called_exception_lambda();
    scope_fail_not_called_no_exception();
    scope_fail_not_called_released();

    scope_success_called_no_exception();
    scope_success_called_no_exception_lambda();
    scope_success_not_called_exception();
    scope_success_not_called_released();
    scope_success_exit_throws();

    return hpx::util::report_errors();
}
