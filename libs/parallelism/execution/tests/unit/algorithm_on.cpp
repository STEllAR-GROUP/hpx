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

template <typename S>
auto tag_dispatch(ex::on_t, S&&, scheduler2&& s)
{
    s.tag_dispatch_overload_called = true;
    return scheduler::sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::on(ex::just(),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::on(ex::just(3),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::on(ex::just(custom_type_non_default_constructible{42}),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::on(
            ex::just(custom_type_non_default_constructible_non_copyable{42}),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::on(ex::just(std::string("hello"), 3),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(!scheduler_execute_called);
        HPX_TEST(scheduler_schedule_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        auto s = ex::just(std::string("hello"), 3) |
            ex::on(scheduler{scheduler_schedule_called,
                scheduler_execute_called, tag_dispatch_overload_called});
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    // tag_dispatch overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::on(ex::just(),
            scheduler2{scheduler{scheduler_schedule_called,
                scheduler_execute_called, tag_dispatch_overload_called}});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(tag_dispatch_overload_called);
        HPX_TEST(!scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        std::atomic<bool> tag_dispatch_overload_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::on(error_sender{},
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_dispatch_overload_called});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
        HPX_TEST(!tag_dispatch_overload_called);
        HPX_TEST(!scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    return hpx::util::report_errors();
}
