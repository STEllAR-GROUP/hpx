//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

    template <typename E>
    void set_error(E&&) noexcept
    {
        HPX_TEST(false);
    }

    void set_done() noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&... ts) noexcept
    {
        HPX_INVOKE(f, std::forward<Ts>(ts)...);
        set_value_called = true;
    }
};

template <typename T>
struct custom_type
{
    std::atomic<bool>& called;
    std::decay_t<T> x;
};

template <typename T>
auto tag_invoke(ex::just_t, custom_type<T> c)
{
    c.called = true;
    return ex::just(c.x);
}

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just();
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(3);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s = ex::just(std::string("hello"), 3);
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        custom_type<int> c{tag_invoke_overload_called, 3};
        auto s = ex::just(c);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(tag_invoke_overload_called);
    }

    return hpx::util::report_errors();
}
