//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

template <typename F>
auto tag_invoke(ex::upon_error_t, custom_sender_tag_invoke s, F&&)
{
    s.tag_invoke_overload_called = true;
    return void_sender{};
}

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> upon_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::upon_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr{}(ep);
            upon_error_callback_called = true;
            return 42;
        });

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int>>>(s2);
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
        check_sends_stopped<false>(s2);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(upon_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> upon_error_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::upon_error(std::move(s1), [&](std::exception_ptr ep) {
            check_exception_ptr{}(ep);
            upon_error_callback_called = true;
            return custom_type_non_default_constructible{42};
        });

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(
            s2);
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
        check_sends_stopped<false>(s2);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(upon_error_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> upon_error_callback_not_called{true};
        auto s1 = ex::just(42);
        auto s2 = ex::upon_error(std::move(s1), [&](std::exception_ptr) {
            upon_error_callback_not_called = false;
            return 0;
        });

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(upon_error_callback_not_called);
    }

    {
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_sender_tag_invoke{tag_invoke_overload_called} |
            ex::upon_error([](auto) { return 0; });
        HPX_TEST(tag_invoke_overload_called);
    }

    return hpx::util::report_errors();
}
