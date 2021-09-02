//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>
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
template <typename... Ss>
auto tag_dispatch(ex::when_all_t, custom_sender_tag_dispatch s, Ss&&... ss)
{
    s.tag_dispatch_overload_called = true;
    return ex::when_all(std::forward<Ss>(ss)...);
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42));
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_dispatch(ex::start, os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(
            ex::just(42), ex::just(std::string("hello")), ex::just(3.14));
        auto f = [](int x, std::string y, double z) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, std::string("hello"));
            HPX_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s =
            ex::when_all(ex::just(custom_type_non_default_constructible(42)));
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(
            ex::just(custom_type_non_default_constructible_non_copyable(42)));
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::when_all(
            custom_sender_tag_dispatch{tag_dispatch_overload_called},
            ex::just(42));
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
        HPX_TEST(tag_dispatch_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_typed_sender<double>{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(ex::just(42), error_typed_sender<double>{});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_typed_sender<double>{}, ex::just(42));
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    return hpx::util::report_errors();
}
