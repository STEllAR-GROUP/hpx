//  Copyright (c) 2022 Hartmut Kaiser
//  Copyright (c) 2022 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
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
namespace tt = hpx::this_thread::experimental;

// NOTE: This is not a conforming sync_wait_with_variant implementation.
// It only exists to check that the tag_invoke overload is called.
void tag_invoke(tt::sync_wait_with_variant_t, custom_sender2 s)
{
    s.tag_invoke_overload_called = true;
}

int hpx_main()
{
    // Success path
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        tt::sync_wait_with_variant(custom_sender{
            start_called, connect_called, tag_invoke_overload_called});
        HPX_TEST(start_called);
        HPX_TEST(connect_called);
        HPX_TEST(!tag_invoke_overload_called);
    }
    // sync_wait_with_variant can accept single value senders :
    // assume currently have one tuple
    {
        auto result = ex::just(42) | tt::sync_wait_with_variant();

        auto v = *result;
        static_assert(
            std::is_same_v<decltype(v), hpx::variant<hpx::tuple<int>>>);
        HPX_TEST(hpx::holds_alternative<hpx::tuple<int>>(v));

        auto t = hpx::get<hpx::tuple<int>>(v);
        static_assert(std::is_same_v<decltype(t), hpx::tuple<int>>);

        auto i = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(i), int>);

        HPX_TEST(i == 42);
    }

    {
        auto result = ex::just(3, 4.0) | tt::sync_wait_with_variant();

        auto v = *result;
        static_assert(
            std::is_same_v<decltype(v), hpx::variant<hpx::tuple<int, double>>>);

        auto t = hpx::get<hpx::tuple<int, double>>(v);
        static_assert(std::is_same_v<decltype(t), hpx::tuple<int, double>>);

        auto i = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(i), int>);

        HPX_TEST(i == 3);

        auto j = hpx::get<1>(t);
        static_assert(std::is_same_v<decltype(j), double>);

        HPX_TEST(j == 4.0);
    }

    {
        auto result =
            tt::sync_wait_with_variant(ex::just(3, 4.0, std::string("42")));
        auto v = *result;

        static_assert(std::is_same_v<decltype(v),
            hpx::variant<hpx::tuple<int, double, std::string>>>);

        auto t = hpx::get<hpx::tuple<int, double, std::string>>(v);
        static_assert(
            std::is_same_v<decltype(t), hpx::tuple<int, double, std::string>>);

        auto i = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(i), int>);

        HPX_TEST(i == 3);

        auto j = hpx::get<1>(t);
        static_assert(std::is_same_v<decltype(j), double>);

        HPX_TEST(j == 4.0);

        auto k = hpx::get<2>(t);
        static_assert(std::is_same_v<decltype(k), std::string>);

        HPX_TEST(k == "42");
    }

    {
        auto s1 = ex::just(custom_type_non_default_constructible{42});
        auto result = tt::sync_wait_with_variant(s1);
        auto v = *result;
        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(
            s1);

        auto t = hpx::get<hpx::tuple<custom_type_non_default_constructible>>(v);
        auto p = hpx::get<0>(t);
        static_assert(
            std::is_same_v<decltype(p), custom_type_non_default_constructible>);

        HPX_TEST_EQ(p.x, 42);
    }

    {
        auto result = tt::sync_wait_with_variant(
            ex::just(custom_type_non_default_constructible_non_copyable{42}));
        auto const& v = *result;
        static_assert(std::is_same_v<std::decay_t<decltype(v)>,
            hpx::variant<hpx::tuple<
                custom_type_non_default_constructible_non_copyable>>>);

        auto const& t = hpx::get<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>(v);
        auto const& p = hpx::get<0>(t);
        static_assert(std::is_same_v<std::decay_t<decltype(p)>,
            custom_type_non_default_constructible_non_copyable>);

        HPX_TEST_EQ(p.x, 42);
    }

    // sync_wait_with_variant can accept more than one senders:
    // (accept variant of multi_tuple senders )
    // tests a sender which has two different value types
    // Success path
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        tt::sync_wait_with_variant(custom_sender_multi_tuple{
            start_called, connect_called, tag_invoke_overload_called, true});
        HPX_TEST(start_called);
        HPX_TEST(connect_called);
        HPX_TEST(!tag_invoke_overload_called);
    }

    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        tt::sync_wait_with_variant(custom_sender_multi_tuple{
            start_called, connect_called, tag_invoke_overload_called, false});
        HPX_TEST(start_called);
        HPX_TEST(connect_called);
        HPX_TEST(!tag_invoke_overload_called);
    }

    {
        auto sd = ex::just(3) | ex::let_error([](std::exception_ptr) {
            HPX_TEST(false);
            return ex::just(std::string{"err"});
        });

        auto result = tt::sync_wait_with_variant(std::move(sd));

        // variant
        auto v = *result;
        static_assert(std::is_same_v<decltype(v),
            hpx::variant<hpx::tuple<std::string>, hpx::tuple<int>>>);

        HPX_TEST(hpx::holds_alternative<hpx::tuple<int>>(v));

        // tuple
        auto t = hpx::get<1>(v);
        static_assert(std::is_same_v<decltype(t), hpx::tuple<int>>);

        auto i = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(i), int>);

        HPX_TEST_EQ(i, 3);
    }

    {
        auto s1 = ex::just(custom_type_non_default_constructible{42});
        auto s2 = ex::let_value(std::move(s1),
            [](custom_type_non_default_constructible const& value) {
                HPX_TEST_EQ(value.x, 42);
                return ex::just(std::to_string(value.x));
            });

        auto result = tt::sync_wait_with_variant(std::move(s2));

        // variant
        auto v = *result;
        static_assert(
            std::is_same_v<decltype(v), hpx::variant<hpx::tuple<std::string>>>);

        // tuple
        auto t = hpx::get<0>(v);
        static_assert(std::is_same_v<decltype(t), hpx::tuple<std::string>>);

        auto j = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(j), std::string>);

        HPX_TEST_EQ(j, std::string("42"));
    }

    // operator| overload
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_sender{
            start_called, connect_called, tag_invoke_overload_called} |
            tt::sync_wait_with_variant();
        HPX_TEST(start_called);
        HPX_TEST(connect_called);
        HPX_TEST(!tag_invoke_overload_called);
    }

    {
        auto result = ex::just(3) | tt::sync_wait_with_variant();

        auto v = *result;
        static_assert(
            std::is_same_v<decltype(v), hpx::variant<hpx::tuple<int>>>);
        HPX_TEST(hpx::holds_alternative<hpx::tuple<int>>(v));

        auto t = hpx::get<hpx::tuple<int>>(v);
        static_assert(std::is_same_v<decltype(t), hpx::tuple<int>>);

        auto i = hpx::get<0>(t);
        static_assert(std::is_same_v<decltype(i), int>);

        HPX_TEST(i == 3);
    }

    // tag_invoke overload
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        tt::sync_wait_with_variant(custom_sender2{custom_sender{
            start_called, connect_called, tag_invoke_overload_called}});
        HPX_TEST(!start_called);
        HPX_TEST(!connect_called);
        HPX_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        bool exception_thrown = false;
        try
        {
            tt::sync_wait_with_variant(error_sender{});
            HPX_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    // cancellation path
    {
        auto result =
            (stopped_sender_with_value_type{} | tt::sync_wait_with_variant());
        HPX_TEST(!result);    // returned optional should be empty
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
