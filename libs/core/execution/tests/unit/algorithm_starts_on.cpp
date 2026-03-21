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
        auto s = ex::starts_on(scheduler, ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
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
        auto s = ex::starts_on(
            scheduler, ex::just(custom_type_non_default_constructible{42}));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
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
        auto s = ex::starts_on(scheduler, ex::just());

        static_assert(ex::is_sender_v<decltype(s)>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        loop.run();
        HPX_TEST(set_value_called);
    }

    return hpx::util::report_errors();
}
