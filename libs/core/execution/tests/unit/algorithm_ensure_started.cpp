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
template <typename Allocator = hpx::util::internal_allocator<>>
auto tag_dispatch(ex::ensure_started_t, custom_sender_tag_dispatch s,
    Allocator const& = Allocator{})
{
    s.tag_dispatch_overload_called = true;
    return void_sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::transform(void_sender{}, [&]() { started = true; });
        auto s2 = ex::ensure_started(std::move(s1));
        HPX_TEST(started);
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::transform(ex::just(0), [&](int x) {
            started = true;
            return x;
        });
        auto s2 = ex::ensure_started(std::move(s1));
        HPX_TEST(started);
        auto f = [](int x) { HPX_TEST_EQ(x, 0); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 =
            ex::transform(ex::just(custom_type_non_default_constructible{42}),
                [&](custom_type_non_default_constructible x) {
                    started = true;
                    return x;
                });
        auto s2 = ex::ensure_started(std::move(s1));
        HPX_TEST(started);
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> started{false};
        auto s1 = ex::transform(
            ex::just(custom_type_non_default_constructible_non_copyable{42}),
            [&](custom_type_non_default_constructible_non_copyable&& x) {
                started = true;
                return std::move(x);
            });
        auto s2 = ex::ensure_started(std::move(s1));
        HPX_TEST(started);
        auto f = [](auto& x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = void_sender{} | ex::ensure_started();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // tag_dispatch overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = custom_sender_tag_dispatch{tag_dispatch_overload_called} |
            ex::ensure_started();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
        HPX_TEST(tag_dispatch_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = error_sender{} | ex::ensure_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = error_sender{} | ex::ensure_started() | ex::ensure_started() |
            ex::ensure_started();
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // Chained ensure_started calls do not create new shared states
    {
        std::atomic<bool> receiver_set_value_called{false};
        auto s1 = ex::just() | ex::ensure_started();
        auto s2 = ex::ensure_started(s1);
        HPX_TEST_EQ(s1.state, s2.state);
        auto s3 = ex::ensure_started(std::move(s2));
        HPX_TEST_EQ(s1.state, s3.state);
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s3), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
    }

    {
        std::atomic<bool> receiver_set_value_called{false};
        auto s1 = ex::just(42) | ex::ensure_started();
        auto s2 = ex::ensure_started(s1);
        HPX_TEST_EQ(s1.state, s2.state);
        auto s3 = ex::ensure_started(std::move(s2));
        HPX_TEST_EQ(s1.state, s3.state);
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s3), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
    }

    return hpx::util::report_errors();
}
