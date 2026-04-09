//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

int hpx_main()
{
    // upon_error: error is converted to value
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_error(
            ex::just_error(std::make_exception_ptr(std::runtime_error("err"))),
            [](std::exception_ptr) { return 42; });
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_error: void-returning handler
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_error(
            ex::just_error(std::make_exception_ptr(std::runtime_error("err"))),
            [](std::exception_ptr) { /* void */ });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_error: value channel is forwarded unchanged
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_error(
            ex::just(7), [](std::exception_ptr) { return 99; });
        static_assert(ex::is_sender_v<decltype(s)>);

        auto f = [](int x) { HPX_TEST_EQ(x, 7); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_error: handler that throws re-sends error as exception_ptr
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::upon_error(
            ex::just_error(std::make_exception_ptr(std::runtime_error("err"))),
            [](std::exception_ptr) -> int {
                throw std::runtime_error("handler_error");
            });
        static_assert(ex::is_sender_v<decltype(s)>);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // upon_error: pipe operator
    {
        std::atomic<bool> set_value_called{false};
        auto s =
            ex::just_error(std::make_exception_ptr(std::runtime_error("e"))) |
            ex::upon_error([](std::exception_ptr) { return std::string("ok"); });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](std::string x) { HPX_TEST_EQ(x, std::string("ok")); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_error: chained with then
    {
        std::atomic<bool> set_value_called{false};
        auto s =
            ex::just_error(std::make_exception_ptr(std::runtime_error("e"))) |
            ex::upon_error([](std::exception_ptr) { return 1; }) |
            ex::then([](int x) { return x + 1; });
        static_assert(ex::is_sender_v<decltype(s)>);

        auto f = [](int x) { HPX_TEST_EQ(x, 2); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_error: stopped channel is forwarded unchanged
    {
        std::atomic<bool> set_stopped_called{false};
        auto s = ex::upon_error(
            ex::just_stopped(), [](std::exception_ptr) { return 42; });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<true>(s);

        auto r = expect_stopped_receiver{set_stopped_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_stopped_called);
    }

    // upon_error: error_sender (sends error from an otherwise-value sender)
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_error(
            error_sender<>{}, [](std::exception_ptr) { return 10; });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_error_types<hpx::variant<std::exception_ptr>>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 10); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
