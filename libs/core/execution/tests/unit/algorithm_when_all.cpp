//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// Clang V11 ICE's on this test, Clang V8 reports a bogus constexpr problem
#if !defined(HPX_CLANG_VERSION) ||                                             \
    ((HPX_CLANG_VERSION / 10000) != 11 && (HPX_CLANG_VERSION / 10000) != 8)

#include <hpx/modules/async_base.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#if defined(HPX_HAVE_STDEXEC)
// TODO: Figure out why this is necessary for stdexec
// but causes dataflow to be unresolvable without stdexec
#include <hpx/execution/algorithms/when_all.hpp>
#endif

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
auto tag_invoke(ex::when_all_t, custom_sender_tag_invoke s, Ss&&... ss)
{
    s.tag_invoke_overload_called = true;
    return ex::when_all(std::forward<Ss>(ss)...);
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        // everything is noexcept and no errors can be made here
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(
            ex::just(42), ex::just(std::string("hello")), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, std::string, double>>>(
            s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

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
        auto s = ex::when_all(
            ex::just(), ex::just(std::string("hello")), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<std::string, double>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](std::string y, double z) {
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
        auto s = ex::when_all(ex::just(42), ex::just(), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, double>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](int x, double z) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(z, 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(), ex::just(), ex::just());

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = []() {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(
            ex::just(42), ex::just(std::string("hello")), ex::just());

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, std::string>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](int x, std::string y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, std::string("hello"));
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::when_all(ex::just(42, std::string("hello"), 3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, std::string, double>>>(
            s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

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
            ex::when_all(ex::just(42, std::string("hello")), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, std::string, double>>>(
            s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

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
        auto s = ex::when_all(ex::just(), ex::just(42, std::string("hello")),
            ex::just(), ex::just(3.14), ex::just());

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif
        check_value_types<hpx::variant<hpx::tuple<int, std::string, double>>>(
            s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

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

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif
        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

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

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif
        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::when_all(
            custom_sender_tag_invoke{tag_invoke_overload_called}, ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif
        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
        HPX_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_typed_sender<double>{});

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif
        check_value_types<hpx::variant<hpx::tuple<double>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(ex::just(42), error_typed_sender<double>{});

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int, double>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::when_all(error_typed_sender<double>{}, ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<double, int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

#if !defined(HPX_HAVE_STDEXEC)
    // Dataflow success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> then_called{false};
        auto s = hpx::dataflow(
            [&](int x) {
                then_called = true;
                HPX_TEST_EQ(x, 42);
                return x;
            },
            ex::just(42));

        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        tag_invoke(ex::start, os);
        HPX_TEST(then_called);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> then_called{false};
        auto s = hpx::dataflow(
            [&](int x, std::string y, double z) {
                then_called = true;
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                return hpx::make_tuple(x, y, z);
            },
            ex::just(42), ex::just(std::string("hello")), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<
            hpx::variant<hpx::tuple<hpx::tuple<int, std::string, double>>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](hpx::tuple<int, std::string, double> t) {
            HPX_TEST_EQ(t.get<0>(), 42);
            HPX_TEST_EQ(t.get<1>(), std::string("hello"));
            HPX_TEST_EQ(t.get<2>(), 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(then_called);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> then_called{false};
        auto s = hpx::dataflow(
            [&](std::string y, double z) {
                then_called = true;
                HPX_TEST_EQ(y, std::string("hello"));
                HPX_TEST_EQ(z, 3.14);
                return hpx::make_tuple(y, z);
            },
            ex::just(), ex::just(std::string("hello")), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<
            hpx::variant<hpx::tuple<hpx::tuple<std::string, double>>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](hpx::tuple<std::string, double> t) {
            HPX_TEST_EQ(t.get<0>(), std::string("hello"));
            HPX_TEST_EQ(t.get<1>(), 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(then_called);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> then_called{false};
        auto s = hpx::dataflow(
            [&](int x, double z) {
                then_called = true;
                HPX_TEST_EQ(x, 42);
                HPX_TEST_EQ(z, 3.14);
                return hpx::make_tuple(x, z);
            },
            ex::just(42), ex::just(), ex::just(3.14));

        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<hpx::tuple<int, double>>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<true>(s);

        auto f = [](hpx::tuple<int, double> t) {
            HPX_TEST_EQ(t.get<0>(), 42);
            HPX_TEST_EQ(t.get<1>(), 3.14);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(then_called);
        HPX_TEST(set_value_called);
    }
#endif

    return hpx::util::report_errors();
}
#else
int main()
{
    return 0;
}
#endif
