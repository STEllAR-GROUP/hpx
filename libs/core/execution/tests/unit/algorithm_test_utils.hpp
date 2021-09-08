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
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r));
        }
    };

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, void_sender, R&& r)
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
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                hpx::execution::experimental::set_error(
                    std::move(os.r), std::current_exception());
            }
        }
    };

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, error_sender, R&& r)
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
    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        callback_receiver&&, E&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_dispatch(
        hpx::execution::experimental::set_done_t, callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend auto tag_dispatch(hpx::execution::experimental::set_value_t,
        callback_receiver&& r, Ts&&... ts) noexcept
        -> decltype(HPX_INVOKE(f, std::forward<Ts>(ts)...), void())
    {
        HPX_INVOKE(r.f, std::forward<Ts>(ts)...);
        r.set_value_called = true;
    }
};

template <typename F>
struct error_callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;
    bool expect_set_value = false;

    template <typename E>
    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        error_callback_receiver&& r, E&& e) noexcept
    {
        HPX_INVOKE(r.f, std::forward<E>(e));
        r.set_error_called = true;
    }

    friend void tag_dispatch(hpx::execution::experimental::set_done_t,
        error_callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend void tag_dispatch(hpx::execution::experimental::set_value_t,
        error_callback_receiver&& r, Ts&&...) noexcept
    {
        HPX_TEST(r.expect_set_value);
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

        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                hpx::execution::experimental::set_error(
                    std::move(os.r), std::current_exception());
            }
        };
    };

    template <typename R>
    friend auto tag_dispatch(
        hpx::execution::experimental::connect_t, error_typed_sender&&, R&& r)
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
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            hpx::execution::experimental::set_value(std::move(os.r));
        };
    };

    template <typename R>
    friend auto tag_dispatch(
        hpx::execution::experimental::connect_t, custom_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{s.start_called, std::forward<R>(r)};
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
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            hpx::execution::experimental::set_value(
                std::move(os.r), std::move(os.x));
        };
    };

    template <typename R>
    friend auto tag_dispatch(
        hpx::execution::experimental::connect_t, custom_typed_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{
            std::move(s.x), s.start_called, std::forward<R>(r)};
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
    friend void tag_dispatch(
        hpx::execution::experimental::execute_t, scheduler s, F&& f)
    {
        s.execute_called = true;
        HPX_INVOKE(std::forward<F>(f));
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
        struct operation_state
        {
            std::decay_t<R> r;

            friend void tag_dispatch(hpx::execution::experimental::start_t,
                operation_state& os) noexcept
            {
                hpx::execution::experimental::set_value(std::move(os.r));
            };
        };

        template <typename R>
        friend auto tag_dispatch(
            hpx::execution::experimental::connect_t, sender&&, R&& r)
        {
            return operation_state<R>{std::forward<R>(r)};
        }
    };

    friend sender tag_dispatch(
        hpx::execution::experimental::schedule_t, scheduler s)
    {
        s.schedule_called = true;
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
