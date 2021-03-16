//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename T>
auto tag_invoke(ex::just_on_t, scheduler2 s, T&& t)
{
    s.tag_invoke_overload_called = true;
    return ex::just_on(
        std::move(static_cast<scheduler>(s)), std::forward<T>(t));
}

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(
            scheduler{scheduler_execute_called, tag_invoke_overload_called});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(
            scheduler{scheduler_execute_called, tag_invoke_overload_called}, 3);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(
            scheduler{scheduler_execute_called, tag_invoke_overload_called},
            std::string("hello"), 3);
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(scheduler2{scheduler{scheduler_execute_called,
                                 tag_invoke_overload_called}},
            3);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    return hpx::util::report_errors();
}
