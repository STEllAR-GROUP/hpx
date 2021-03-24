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

struct scheduler
{
    std::atomic<bool>& execute_called;

    template <typename F>
    void execute(F&& f) const
    {
        execute_called = true;
        HPX_INVOKE(std::forward<F>(f));
    }

    // The following are only here to make this a valid scheduler. The current
    // implementation makes use of execute, but the implementation can be
    // changed. If that happens the test for which function should be called
    // should also be changed.
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept
        {
            ex::set_value(std::move(r));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        return operation_state<R>{std::forward<R>(r)};
    }

    constexpr void schedule() const {}

    bool operator==(scheduler const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler const&) const noexcept
    {
        return false;
    }
};

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
    auto set_value(Ts&&... ts) noexcept
        -> decltype(HPX_INVOKE(f, std::forward<Ts>(ts)...), void())
    {
        HPX_INVOKE(f, std::forward<Ts>(ts)...);
        set_value_called = true;
    }
};

template <typename T>
struct custom_type
{
    std::atomic<bool>& tag_invoke_overload_called;
    std::decay_t<T> x;
};

template <typename S, typename T>
auto tag_invoke(ex::just_on_t, S&& s, custom_type<T> c)
{
    c.tag_invoke_overload_called = true;
    return ex::just_on(std::forward<S>(s), c.x);
}

int main()
{
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(scheduler{scheduler_execute_called});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(scheduler{scheduler_execute_called}, 3);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::just_on(
            scheduler{scheduler_execute_called}, std::string("hello"), 3);
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        custom_type<int> c{tag_invoke_overload_called, 3};
        auto s = ex::just_on(scheduler{scheduler_execute_called}, c);
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    return hpx::util::report_errors();
}
