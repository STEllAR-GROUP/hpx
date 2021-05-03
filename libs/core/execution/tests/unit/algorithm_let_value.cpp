//  Copyright (c) 2021 ETH Zurich
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

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename F>
auto tag_dispatch(ex::let_value_t, custom_sender_tag_dispatch s, F&&)
{
    s.tag_dispatch_overload_called = true;
    return void_sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 = void_sender{};
        auto s2 = ex::let_value(std::move(s1), [&]() {
            let_value_callback_called = true;
            return void_sender();
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 = ex::just(42);
        auto s2 = ex::let_value(std::move(s1), [&](int& x) {
            HPX_TEST_EQ(x, 42);
            let_value_callback_called = true;
            return ex::just(x);
        });
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 = ex::just(custom_type_non_default_constructible{42});
        auto s2 = ex::let_value(std::move(s1), [&](auto& x) {
            HPX_TEST_EQ(x.x, 42);
            let_value_callback_called = true;
            return ex::just(x);
        });
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 =
            ex::just(custom_type_non_default_constructible_non_copyable{42});
        auto s2 = ex::let_value(std::move(s1), [&](auto& x) {
            HPX_TEST_EQ(x.x, 42);
            let_value_callback_called = true;
            return ex::just(std::move(x));
        });
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s = void_sender{} | ex::let_value([&]() {
            let_value_callback_called = true;
            return void_sender();
        });
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s = ex::just(42) | ex::let_value([&](int& x) {
            HPX_TEST_EQ(x, 42);
            let_value_callback_called = true;
            return ex::just(x);
        });
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(let_value_callback_called);
    }

    // tag_dispatch overload
    {
        std::atomic<bool> tag_dispatch_overload_called{false};
        custom_sender_tag_dispatch{tag_dispatch_overload_called} |
            ex::let_value([]() { return ex::just(); });
        HPX_TEST(tag_dispatch_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_value(std::move(s1), [&]() {
            let_value_callback_called = true;
            return void_sender();
        });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
        HPX_TEST(!let_value_callback_called);
    }

    {
        // Can have multiple value types (i.e. different for the predecessor
        // sender, and the one produced by the factory), as long as the receiver
        // connected to the let_value sender can handle them. In the test below
        // value_types for the let_value sender is Variant<Tuple<int>, Tuple<>>.
        std::atomic<bool> set_error_called{false};
        std::atomic<bool> let_value_callback_called{false};
        auto s1 = error_sender{};
        auto s2 = ex::let_value(std::move(s1), [&]() {
            let_value_callback_called = true;
            return ex::just(42);
        });
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
        HPX_TEST(!let_value_callback_called);
    }

    return hpx::util::report_errors();
}
