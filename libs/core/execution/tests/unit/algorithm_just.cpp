//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/modules/datastructures.hpp>

#ifdef HPX_HAVE_STDEXEC
#include <hpx/execution/algorithms/just.hpp>
#else
#include <hpx/modules/execution.hpp>
#endif
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just();

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
#ifdef HPX_HAVE_STDEXEC
        // the just sender does not produce errors in STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(3);

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        int x = 3;
        auto s = ex::just(x);

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

#ifdef HPX_HAVE_STDEXEC
        // the just sender decay-copies the value in STDEXEC instead of
        // storing a reference
        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
        check_error_types<hpx::variant<>>(s);
#else
        check_value_types<hpx::variant<hpx::tuple<int&>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(custom_type_non_default_constructible{42});

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        custom_type_non_default_constructible x{42};
        auto s = ex::just(x);

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

#ifdef HPX_HAVE_STDEXEC
        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(
            s);
        check_error_types<hpx::variant<>>(s);
#else
        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible&>>>(
            s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s =
            ex::just(custom_type_non_default_constructible_non_copyable{42});

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        custom_type_non_default_constructible_non_copyable x{42};
        auto s = ex::just(std::move(x));

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(std::string("hello"), 3);

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<std::string, int>>>(s);
#ifdef HPX_HAVE_STDEXEC
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::string str{"hello"};
        int x = 3;
        auto s = ex::just(str, x);

        static_assert(ex::is_sender_v<decltype(s)>);
#ifdef HPX_HAVE_STDEXEC        
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif        

#ifdef HPX_HAVE_STDEXEC
        check_value_types<hpx::variant<hpx::tuple<std::string, int>>>(s);
        check_error_types<hpx::variant<>>(s);
#else
        check_value_types<hpx::variant<hpx::tuple<std::string&, int&>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<false>(s);

        auto f = [](std::string str, int x) {
            HPX_TEST_EQ(str, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    return hpx::util::report_errors();
}
