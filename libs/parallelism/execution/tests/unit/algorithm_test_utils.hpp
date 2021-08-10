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

#pragma once

struct void_sender
{
    template <template <typename...> class Tuple,
        template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept
        {
            hpx::execution::experimental::set_value(std::move(r));
        }
    };

    template <typename R>
    operation_state<R> connect(R&& r)
    {
        return {std::forward<R>(r)};
    }
};

struct error_sender
{
    template <template <typename...> class Tuple,
        template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
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
                hpx::execution::experimental::set_error(
                    std::move(r), std::current_exception());
            }
        }
    };

    template <typename R>
    operation_state<R> connect(R&& r)
    {
        return {std::forward<R>(r)};
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

template <typename F>
struct error_callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;
    bool expect_set_value = false;

    template <typename E>
    void set_error(E&& e) noexcept
    {
        HPX_INVOKE(f, std::forward<E>(e));
        set_error_called = true;
    }

    void set_done() noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    void set_value(Ts&&...) noexcept
    {
        HPX_TEST(expect_set_value);
    }
};

template <typename F>
struct void_callback_helper
{
    std::decay_t<F> f;

    // This overload is only used to satisfy tests that have a predecessor that
    // can send void, but never does in practice.
    void operator()() const
    {
        HPX_TEST(false);
    }

    template <typename T>
    void operator()(T&& t)
    {
        HPX_INVOKE(std::move(f), std::forward<T>(t));
    }
};

template <typename T>
struct error_typed_sender
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<T>>;

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
                hpx::execution::experimental::set_error(
                    std::move(r), std::current_exception());
            }
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        return operation_state<R>{std::forward<R>(r)};
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

struct custom_sender_tag_dispatch
{
    std::atomic<bool>& tag_dispatch_overload_called;

    template <template <typename...> class Tuple,
        template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = Variant<>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept
        {
            hpx::execution::experimental::set_value(std::move(r));
        }
    };

    template <typename R>
    operation_state<R> connect(R&& r)
    {
        return {std::forward<R>(r)};
    }
};

struct custom_sender
{
    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_dispatch_overload_called;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        void start() noexcept
        {
            start_called = true;
            hpx::execution::experimental::set_value(std::move(r));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        connect_called = true;
        return operation_state<R>{start_called, std::forward<R>(r)};
    }
};

template <typename T>
struct custom_typed_sender
{
    std::decay_t<T> x;

    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_dispatch_overload_called;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<T> x;
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        void start() noexcept
        {
            start_called = true;
            hpx::execution::experimental::set_value(std::move(r), std::move(x));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        connect_called = true;
        return operation_state<R>{
            std::move(x), start_called, std::forward<R>(r)};
    }
};

struct custom_sender2 : custom_sender
{
    explicit custom_sender2(custom_sender s)
      : custom_sender(std::move(s))
    {
    }
};

template <typename T>
struct custom_type
{
    std::atomic<bool>& tag_dispatch_overload_called;
    std::decay_t<T> x;
};

struct custom_type_non_default_constructible
{
    int x;
    custom_type_non_default_constructible() = delete;
    explicit custom_type_non_default_constructible(int x)
      : x(x){};
    custom_type_non_default_constructible(
        custom_type_non_default_constructible&&) = default;
    custom_type_non_default_constructible& operator=(
        custom_type_non_default_constructible&&) = default;
    custom_type_non_default_constructible(
        custom_type_non_default_constructible const&) = default;
    custom_type_non_default_constructible& operator=(
        custom_type_non_default_constructible const&) = default;
};

struct custom_type_non_default_constructible_non_copyable
{
    int x;
    custom_type_non_default_constructible_non_copyable() = delete;
    explicit custom_type_non_default_constructible_non_copyable(int x)
      : x(x){};
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable const&) = delete;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable const&) = delete;
};

struct scheduler
{
    std::atomic<bool>& schedule_called;
    std::atomic<bool>& execute_called;
    std::atomic<bool>& tag_dispatch_overload_called;

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
            hpx::execution::experimental::set_value(std::move(r));
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

    sender schedule()
    {
        schedule_called = true;
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
