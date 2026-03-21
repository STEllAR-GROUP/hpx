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
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

int main()
{
    {
        ex::run_loop loop;
        auto scheduler = loop.get_scheduler();

        std::atomic<bool> set_value_called{false};
        auto s = ex::transfer_when_all_with_variant(
            scheduler, ex::just(42), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        auto f = [](auto v1, auto v2) {
            HPX_TEST(hpx::holds_alternative<hpx::tuple<int>>(v1));
            HPX_TEST(hpx::holds_alternative<hpx::tuple<double>>(v2));
            HPX_TEST_EQ(hpx::get<0>(hpx::get<hpx::tuple<int>>(v1)), 42);
            HPX_TEST_EQ(hpx::get<0>(hpx::get<hpx::tuple<double>>(v2)), 3.14);
        };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        loop.run();
        HPX_TEST(set_value_called);
    }

    {
        ex::run_loop loop;
        auto scheduler = loop.get_scheduler();

        std::atomic<bool> set_value_called{false};
        auto s = ex::transfer_when_all_with_variant(scheduler,
            ex::just(custom_type_non_default_constructible{42}),
            ex::just(custom_type_non_default_constructible{100}));

        static_assert(ex::is_sender_v<decltype(s)>);

        auto f = [](auto v1, auto v2) {
            HPX_TEST(hpx::holds_alternative<
                hpx::tuple<custom_type_non_default_constructible>>(v1));
            HPX_TEST(hpx::holds_alternative<
                hpx::tuple<custom_type_non_default_constructible>>(v2));
            HPX_TEST_EQ(
                hpx::get<0>(hpx::get<
                    hpx::tuple<custom_type_non_default_constructible>>(v1))
                    .x,
                42);
            HPX_TEST_EQ(
                hpx::get<0>(hpx::get<
                    hpx::tuple<custom_type_non_default_constructible>>(v2))
                    .x,
                100);
        };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        loop.run();
        HPX_TEST(set_value_called);
    }

    {
        ex::run_loop loop;
        auto scheduler = loop.get_scheduler();

        std::atomic<bool> set_value_called{false};
        auto s =
            ex::transfer_when_all_with_variant(scheduler, ex::just(), ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);

        auto f = [](auto v1, auto v2) {
            HPX_TEST(hpx::holds_alternative<hpx::tuple<>>(v1));
            HPX_TEST(hpx::holds_alternative<hpx::tuple<int>>(v2));
            HPX_TEST_EQ(hpx::get<0>(hpx::get<hpx::tuple<int>>(v2)), 42);
        };
        auto r = callback_receiver<void_callback_helper<decltype(f)>>{
            void_callback_helper<decltype(f)>{f}, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        loop.run();
        HPX_TEST(set_value_called);
    }

    return hpx::util::report_errors();
}
