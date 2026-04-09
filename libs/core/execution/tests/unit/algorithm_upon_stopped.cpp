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
    // upon_stopped: stopped is converted to value
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_stopped(ex::just_stopped(), [] { return 42; });
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_stopped: void-returning handler
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_stopped(ex::just_stopped(), [] { /* void */ });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_stopped: value channel is forwarded unchanged
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::upon_stopped(ex::just(7), [] { return 99; });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 7); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_stopped: error channel is forwarded unchanged
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::upon_stopped(
            ex::just_error(std::make_exception_ptr(std::runtime_error("err"))),
            [] { return 42; });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // upon_stopped: handler that throws re-sends error as exception_ptr
    {
        std::atomic<bool> set_error_called{false};
        auto s = ex::upon_stopped(ex::just_stopped(), []() -> int {
            throw std::runtime_error("handler_error");
        });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // upon_stopped: pipe operator
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just_stopped() |
            ex::upon_stopped([] { return std::string("recovered"); });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto f = [](std::string x) {
            HPX_TEST_EQ(x, std::string("recovered"));
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_stopped: chained with then
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just_stopped() |
            ex::upon_stopped([] { return 1; }) |
            ex::then([](int x) { return x + 1; });
        static_assert(ex::is_sender_v<decltype(s)>);

        auto f = [](int x) { HPX_TEST_EQ(x, 2); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // upon_stopped: stopped_sender_with_value_type (sends stopped even though
    // it also advertises a value type)
    {
        std::atomic<bool> set_value_called{false};
        auto s =
            ex::upon_stopped(stopped_sender_with_value_type{}, [] { return 5; });
        static_assert(ex::is_sender_v<decltype(s)>);

        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 5); };
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
