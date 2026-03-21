//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/optional.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

template <typename T>
using optional = hpx::optional<T>;

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        auto s1 = stopped_sender{};
        auto s2 = ex::stopped_as_optional(std::move(s1));

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<optional<>>>>(s2);
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
        check_sends_stopped<false>(s2);

        auto f = [](optional<> opt) { HPX_TEST(!opt.has_value()); };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(42);
        auto s2 = ex::stopped_as_optional(std::move(s1));

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<optional<int>>>>(s2);
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
        check_sends_stopped<false>(s2);

        auto f = [](optional<int> opt) {
            HPX_TEST(opt.has_value());
            HPX_TEST_EQ(*opt, 42);
        };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(42, 3.14);
        auto s2 = ex::stopped_as_optional(std::move(s1));

        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<optional<int, double>>>>(s2);
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
        check_sends_stopped<false>(s2);

        auto f = [](optional<int, double> opt) {
            HPX_TEST(opt.has_value());
            HPX_TEST_EQ(hpx::get<0>(*opt), 42);
            HPX_TEST_EQ(hpx::get<1>(*opt), 3.14);
        };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    return hpx::util::report_errors();
}
