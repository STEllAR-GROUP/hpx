//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#pragma once

// Check that the value_types of a sender matches the expected type
template <typename ExpectedValType,
    typename Env = hpx::execution::experimental::empty_env, typename Sender>
inline void check_value_types(Sender&&)
{
    namespace ex = hpx::execution::experimental;
    using value_types = ex::value_types_of_t<Sender, Env, hpx::tuple,
        ex::detail::unique_variant<hpx::variant>::template apply>;
    static_assert(std::is_same_v<value_types, ExpectedValType>);
}

// Check that the error_types of a sender matches the expected type
template <typename ExpectedErrorType,
    typename Env = hpx::execution::experimental::empty_env, typename Sender>
inline void check_error_types(Sender&&)
{
    namespace ex = hpx::execution::experimental;
    using error_types = ex::error_types_of_t<Sender, Env,
        ex::detail::unique_variant<hpx::variant>::template apply>;
    static_assert(std::is_same_v<error_types, ExpectedErrorType>);
}

//! Check that the sends_stopped of a sender matches the expected value
template <bool Expected, typename Env = hpx::execution::experimental::empty_env,
    typename Sender>
inline void check_sends_stopped(Sender&&)
{
    constexpr bool sends_stopped =
        hpx::execution::experimental::completion_signatures_of_t<Sender,
            Env>::sends_stopped;
    static_assert(sends_stopped == Expected);
}

struct void_sender
{
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(std::move(os.r));
        }
    };

    template <typename R>
    friend operation_state<R> tag_invoke(
        hpx::execution::experimental::connect_t, void_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        void_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t()>;
};

template <typename... Ts>
struct error_sender
{
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        friend void tag_invoke(
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
    friend operation_state<R> tag_invoke(
        hpx::execution::experimental::connect_t, error_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        error_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(Ts...),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;
};

struct stopped_sender
{
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_stopped(std::move(os.r));
        }
    };

    template <typename R>
    friend operation_state<R> tag_invoke(
        hpx::execution::experimental::connect_t, stopped_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        stopped_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_stopped_t()>;
};

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

    template <typename E>
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        callback_receiver&&, E&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_invoke(hpx::execution::experimental::set_stopped_t,
        callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend auto tag_invoke(hpx::execution::experimental::set_value_t,
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
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        error_callback_receiver&& r, E&& e) noexcept
    {
        HPX_INVOKE(r.f, std::forward<E>(e));
        r.set_error_called = true;
    }

    friend void tag_invoke(hpx::execution::experimental::set_stopped_t,
        error_callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend void tag_invoke(hpx::execution::experimental::set_value_t,
        error_callback_receiver&& r, Ts&&...) noexcept
    {
        HPX_TEST(r.expect_set_value);
    }
};

struct expect_stopped_receiver
{
    std::atomic<bool>& set_stopped_called;

    template <typename... Ts>
    friend void tag_invoke(hpx::execution::experimental::set_value_t,
        expect_stopped_receiver&&, Ts...) noexcept
    {
        HPX_TEST(false);    // should not be called
    }
    friend void tag_invoke(hpx::execution::experimental::set_stopped_t,
        expect_stopped_receiver&& r) noexcept
    {
        r.set_stopped_called = true;
    }
    friend void tag_invoke(hpx::execution::experimental::set_error_t,
        expect_stopped_receiver&&, std::exception_ptr) noexcept
    {
        HPX_TEST(false);    // should not be called
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
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;

        friend void tag_invoke(
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
    friend auto tag_invoke(
        hpx::execution::experimental::connect_t, error_typed_sender&&, R&& r)
    {
        return operation_state<R>{std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        error_typed_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(T),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;
};

struct check_exception_ptr
{
    void operator()(std::exception_ptr eptr) const
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
        }
    }

    void operator()(std::runtime_error const& e) const
    {
        HPX_TEST_EQ(std::string(e.what()), std::string("error"));
    }
};

struct custom_sender_tag_invoke
{
    std::atomic<bool>& tag_invoke_overload_called;

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
    friend operation_state<R> tag_invoke(
        hpx::execution::experimental::connect_t, custom_sender_tag_invoke&&,
        R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        custom_sender_tag_invoke const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t()>;
};

struct custom_sender
{
    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_stopped = false;

    template <typename R>
    struct operation_state
    {
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            hpx::execution::experimental::set_value(std::move(os.r));
        };
    };

    template <typename R>
    friend auto tag_invoke(
        hpx::execution::experimental::connect_t, custom_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{s.start_called, std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        custom_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;
};

template <typename T>
struct custom_typed_sender
{
    std::decay_t<T> x;

    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    template <typename R>
    struct operation_state
    {
        std::decay_t<T> x;
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            hpx::execution::experimental::set_value(
                std::move(os.r), std::move(os.x));
        };
    };

    template <typename R>
    friend auto tag_invoke(
        hpx::execution::experimental::connect_t, custom_typed_sender&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{
            std::move(s.x), s.start_called, std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        custom_typed_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(T),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;
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
    std::atomic<bool>& tag_invoke_overload_called;
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
    std::atomic<bool>& tag_invoke_overload_called;

    template <typename F>
    friend void tag_invoke(
        hpx::execution::experimental::execute_t, scheduler s, F&& f)
    {
        s.execute_called = true;
        HPX_INVOKE(std::forward<F>(f), );
    }

    struct sender
    {
        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;

            friend void tag_invoke(hpx::execution::experimental::start_t,
                operation_state& os) noexcept
            {
                hpx::execution::experimental::set_value(std::move(os.r));
            };
        };

        template <typename R>
        friend auto tag_invoke(
            hpx::execution::experimental::connect_t, sender&&, R&& r)
        {
            return operation_state<R>{std::forward<R>(r)};
        }

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            sender const&, Env)
            -> hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr)>;
    };

    friend sender tag_invoke(
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
