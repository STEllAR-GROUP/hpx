//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

struct scheduler
{
    std::atomic<bool>& execute_called;
    std::atomic<bool>& tag_invoke_on_called;

    template <typename F>
    void execute(F&& f) const
    {
        execute_called = true;
        HPX_INVOKE(std::forward<F>(f));
    }

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

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

    struct sender
    {
        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        auto connect(R&& r) &&
        {
            return operation_state<R>{std::forward<R>(r)};
        }
    };

    constexpr sender schedule() const
    {
        return {};
    }

    bool operator==(scheduler const&) const noexcept
    {
        return true;
    }

    bool operator!=(scheduler const&) const noexcept
    {
        return false;
    }
};

struct scheduler2 : scheduler
{
    explicit scheduler2(scheduler s)
      : scheduler(std::move(s))
    {
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

struct error_sender
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                ex::set_error(std::move(r), std::current_exception());
            }
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        return operation_state<R>{std::forward<R>(r)};
    }
};

template <typename F>
struct error_callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;

    template <typename E>
    void set_error(E&&) noexcept
    {
        set_error_called = true;
    }

    void set_done() noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) noexcept
    {
        HPX_TEST(false);
    }
};

void check_exception_ptr(std::exception_ptr eptr)
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (const std::runtime_error& e)
    {
        HPX_TEST_EQ(std::string(e.what()), std::string("error"));
    }
};

template <typename S>
auto tag_invoke(ex::on_t, S&&, scheduler2&& s)
{
    s.tag_invoke_on_called = true;
    return scheduler::sender{};
}

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::on(ex::just(),
            scheduler{scheduler_execute_called, tag_invoke_overload_called});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::on(ex::just(3),
            scheduler{scheduler_execute_called, tag_invoke_overload_called});
        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::on(ex::just(std::string("hello"), 3),
            scheduler{scheduler_execute_called, tag_invoke_overload_called});
        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(s, r));
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::on(ex::just(),
            scheduler2{scheduler{
                scheduler_execute_called, tag_invoke_overload_called}});
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        ex::start(ex::connect(std::move(s), r));
        HPX_TEST(set_value_called);
        HPX_TEST(tag_invoke_overload_called);
        HPX_TEST(!scheduler_execute_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::on(error_sender{},
            scheduler{scheduler_execute_called, tag_invoke_overload_called});
        auto r = error_callback_receiver<decltype(check_exception_ptr)>{
            check_exception_ptr, set_error_called};
        ex::start(ex::connect(std::move(s), r));
        HPX_TEST(set_error_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(!scheduler_execute_called);
    }

    return hpx::util::report_errors();
}
