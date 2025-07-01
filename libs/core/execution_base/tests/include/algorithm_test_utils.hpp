//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution/algorithms/execute.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#pragma once

#if defined(HPX_HAVE_STDEXEC)
template <typename Scheduler>
struct env_with_scheduler
{
    template <typename CPO>
    friend Scheduler tag_invoke(
        hpx::execution::experimental::get_completion_scheduler_t<CPO>,
        env_with_scheduler const&) noexcept
    {
        return {};
    }
};
#endif

// Check that the value_types of a sender matches the expected type
template <typename ExpectedValType,
    typename Env = hpx::execution::experimental::empty_env, typename Sender>
inline void check_value_types(Sender&&)
{
    // STDEXEC R7:
    // Passing check_value_types(s) with s being a `custom_sender` lvalue will
    // result in Sender being custom_sender&. This is problematic because the
    // sender<S> concept requires constructible_from<std::remove_cvref_t<S>, S>,
    // so for S = custom_sender the sender concept requires the following to be
    // well-defined:
    // custom_sender s(std::declval<custom_sender>())
    //                      ^^^-- custom_sender&&
    // Whereas for some_sender& the concept requires
    // custom_sender s(std::declval<custom_sender&>())
    //                      ^^^-- custom_sender&

    using UnderlyingSender = std::remove_reference_t<Sender>;
    namespace ex = hpx::execution::experimental;
    using value_types = ex::value_types_of_t<UnderlyingSender, Env, hpx::tuple,
        ex::detail::unique_variant<hpx::variant>::template apply>;
    static_assert(std::is_same_v<value_types, ExpectedValType>);
}

// Check that the error_types of a sender matches the expected type
template <typename ExpectedErrorType,
    typename Env = hpx::execution::experimental::empty_env, typename Sender>
inline void check_error_types(Sender&&)
{
    // See check_value_types
    using UnderlyingSender = std::remove_reference_t<Sender>;

    namespace ex = hpx::execution::experimental;
    using error_types = ex::error_types_of_t<UnderlyingSender, Env,
        ex::detail::unique_variant<hpx::variant>::template apply>;
    static_assert(std::is_same_v<error_types, ExpectedErrorType>);
}

//! Check that the sends_stopped of a sender matches the expected value
template <bool Expected, typename Env = hpx::execution::experimental::empty_env,
    typename Sender>
inline void check_sends_stopped(Sender&&)
{
#if defined(HPX_HAVE_STDEXEC)
    // See check_value_types
    using UnderlyingSender = std::remove_reference_t<Sender>;
    constexpr bool sends_stopped =
        hpx::execution::experimental::sends_stopped<UnderlyingSender, Env>;
#else
    constexpr bool sends_stopped =
        hpx::execution::experimental::completion_signatures_of_t<Sender,
            Env>::sends_stopped;
#endif
    static_assert(sends_stopped == Expected);
}

struct void_sender
{
    using is_sender = void;
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

    //    using completion_signatures = hpx::execution::experimental::
    //    completion_signatures<hpx::execution::experimental::set_value_t()>;

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        void_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t()>
    {
        return {};
    }
};

template <typename... Ts>
struct error_sender
{
#if defined(HPX_HAVE_STDEXEC)
    using sender_concept = hpx::execution::experimental::sender_t;
#endif

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::detail::try_catch_exception_ptr(
                []() { throw std::runtime_error("error"); },
                [&](std::exception_ptr ep) {
                    hpx::execution::experimental::set_error(
                        std::move(os.r), std::move(ep));
                });
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
            hpx::execution::experimental::set_error_t(std::exception_ptr)>
    {
        return {};
    }
};

struct stopped_sender
{
    struct is_sender
    {
    };
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

struct stopped_sender_with_value_type
{
    using is_sender = void;

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
        hpx::execution::experimental::connect_t, stopped_sender_with_value_type,
        R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        stopped_sender_with_value_type const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_stopped_t(),
            hpx::execution::experimental::set_value_t()>;
};

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

#if defined(HPX_HAVE_STDEXEC)
    using is_receiver = void;
#else
    struct is_receiver
    {
    };
#endif

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

    // clang-format off
    template <typename... Ts>
    friend auto tag_invoke(hpx::execution::experimental::set_value_t,
        callback_receiver&& r,
        Ts&&... ts) noexcept -> decltype(HPX_INVOKE(f, std::forward<Ts>(ts)...),
                                 void())
    {
        HPX_INVOKE(r.f, std::forward<Ts>(ts)...);
        r.set_value_called = true;
    }
    // clang-format on
};

template <typename F>
struct error_callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_error_called;
    bool expect_set_value = false;

#if defined(HPX_HAVE_STDEXEC)
    using is_receiver = void;
#endif

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
#if defined(HPX_HAVE_STDEXEC)
    using is_receiver = void;
#endif

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
    struct is_sender
    {
    };
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;

        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::detail::try_catch_exception_ptr(
                []() { throw std::runtime_error("error"); },
                [&](std::exception_ptr ep) {
                    hpx::execution::experimental::set_error(
                        std::move(os.r), std::move(ep));
                });
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

template <typename T>
struct const_reference_sender
{
    struct is_sender
    {
    };
    std::reference_wrapper<std::decay_t<T>> x;

    template <typename R>
    struct operation_state
    {
        std::reference_wrapper<std::decay_t<T>> const x;
        std::decay_t<R> r;

        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::execution::experimental::set_value(
                std::move(os.r), os.x.get());
        };
    };

    template <typename R>
    friend auto tag_invoke(hpx::execution::experimental::connect_t,
        const_reference_sender&& s, R&& r)
    {
        return operation_state<R>{std::move(s.x), std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        const_reference_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(std::decay_t<T>&),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;
};

struct const_reference_error_sender
{
    struct is_sender
    {
    };
    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;

        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            auto const e = std::make_exception_ptr(std::runtime_error("error"));
            hpx::execution::experimental::set_error(std::move(os.r), e);
        }
    };

    template <typename R>
    friend operation_state<R> tag_invoke(
        hpx::execution::experimental::connect_t, const_reference_error_sender,
        R&& r)
    {
        return {std::forward<R>(r)};
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        const_reference_error_sender const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(),
            hpx::execution::experimental::set_error_t(
                std::exception_ptr const&)>;
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
    using is_sender = void;

    std::atomic<bool>& tag_invoke_overload_called;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;

        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start();
        }

        void start() & noexcept
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
    using is_sender = void;
    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        custom_sender const&, Env&&)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;

#if !defined(HPX_HAVE_STDEXEC)
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_stopped = false;
#endif

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
};

struct custom_sender_multi_tuple
{
    using is_sender = void;
    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_overload_called;

    bool expect_set_value = true;

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        custom_sender_multi_tuple const&, Env&&)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t(int),
            hpx::execution::experimental::set_value_t(std::string),
            hpx::execution::experimental::set_error_t(std::exception_ptr)>;

#if !defined(HPX_HAVE_STDEXEC)
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_stopped = false;
#endif

    template <typename R>
    struct operation_state
    {
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        bool expect_set_value = true;
        friend void tag_invoke(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_called = true;
            if (os.expect_set_value)
            {
                hpx::execution::experimental::set_value(std::move(os.r), 3);
            }
            else
            {
                hpx::execution::experimental::set_value(std::move(os.r), "err");
            }
        };
    };

    template <typename R>
    friend auto tag_invoke(hpx::execution::experimental::connect_t,
        custom_sender_multi_tuple&& s, R&& r)
    {
        s.connect_called = true;
        return operation_state<R>{s.start_called, std::forward<R>(r)};
    }
};

template <typename T>
struct custom_typed_sender
{
    using is_sender = void;
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
        custom_typed_sender const&, Env&&)
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
      : x(x) {};
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
      : x(x) {};
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable&&) = default;
    custom_type_non_default_constructible_non_copyable(
        custom_type_non_default_constructible_non_copyable const&) = delete;
    custom_type_non_default_constructible_non_copyable& operator=(
        custom_type_non_default_constructible_non_copyable const&) = delete;
};

template <typename Derived>
struct example_scheduler_template
{
    std::atomic<bool>& schedule_called;
    std::atomic<bool>& execute_called;
    std::atomic<bool>& tag_invoke_overload_called;

    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    example_scheduler_template(std::atomic<bool>& schedule_called,
        std::atomic<bool>& execute_called,
        std::atomic<bool>& tag_invoke_overload_called)
      : schedule_called(schedule_called)
      , execute_called(execute_called)
      , tag_invoke_overload_called(tag_invoke_overload_called)
    {
    }

    template <typename F>
    friend void tag_invoke(hpx::execution::experimental::execute_t,
        example_scheduler_template s, F&& f)
    {
        s.execute_called = true;
        HPX_INVOKE(std::forward<F>(f), );
    }

    struct my_sender
    {
        using is_sender = void;
#if defined(HPX_HAVE_STDEXEC)
        friend env_with_scheduler<std::conditional_t<std::is_void_v<Derived>,
            example_scheduler_template, Derived>>
        tag_invoke(
            hpx::execution::experimental::get_env_t, my_sender const&) noexcept
        {
            return {};
        }
#endif

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
            hpx::execution::experimental::connect_t, my_sender&&, R&& r)
        {
            return operation_state<R>{std::forward<R>(r)};
        }

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            my_sender const&, Env)
            -> hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr)>;
    };

    friend my_sender tag_invoke(
        hpx::execution::experimental::schedule_t, example_scheduler_template s)
    {
        s.schedule_called = true;
        return {};
    }

    bool operator==(example_scheduler_template const&) const noexcept
    {
        return true;
    }

    bool operator!=(example_scheduler_template const&) const noexcept
    {
        return false;
    }

    template <typename D>
    example_scheduler_template(example_scheduler_template<D> const& other)
      : schedule_called(other.schedule_called)
      , execute_called(other.execute_called)
      , tag_invoke_overload_called(other.tag_invoke_overload_called)
    {
    }
};

using example_scheduler = example_scheduler_template<void>;
#if defined(HPX_HAVE_STDEXEC)
struct scheduler2 : example_scheduler_template<scheduler2>
{
    explicit scheduler2(example_scheduler s)
      : example_scheduler_template<scheduler2>(std::move(s))
    {
    }
};
#else
struct scheduler2 : example_scheduler
{
    explicit scheduler2(example_scheduler s)
      : example_scheduler(std::move(s))
    {
    }
};
#endif

namespace tag_namespace {

    inline constexpr struct my_tag_t
    {
        template <typename Sender>
        auto operator()(Sender&& sender) const
        {
            return hpx::functional::tag_invoke(
                *this, std::forward<Sender>(sender));
        }

        struct wrapper
        {
            wrapper(my_tag_t) {}
        };

        // This overload should be chosen by test_adl_isolation below. We make
        // sure this is a worse match than the one in my_namespace by requiring
        // a conversion.
        template <typename Sender>
        friend void tag_invoke(wrapper, Sender&&)
        {
        }
    } my_tag{};
}    // namespace tag_namespace

namespace my_namespace {

    // The below types should be used as a template argument for the sender in
    // test_adl_isolation.
    struct my_type
    {
        void operator()() const {}
        void operator()(int) const {}
        void operator()(std::exception_ptr) const {}
    };

    template <int...>
    struct my_scheduler_template
    {
        struct my_sender
        {
#if defined(HPX_HAVE_STDEXEC)
            using sender_concept = hpx::execution::experimental::sender_t;
#endif
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
                hpx::execution::experimental::connect_t, my_sender&&, R&& r)
            {
                return operation_state<R>{std::forward<R>(r)};
            }
#if defined(HPX_HAVE_STDEXEC)
            friend env_with_scheduler<my_scheduler_template> tag_invoke(
                hpx::execution::experimental::get_env_t,
                my_sender const&) noexcept
            {
                return {};
            }
#else
            friend my_scheduler_template tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    hpx::execution::experimental::set_value_t>,
                my_sender const&) noexcept
            {
                return {};
            }
#endif

            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                my_sender const&, Env)
                -> hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_value_t()>;
        };

        friend my_sender tag_invoke(
            hpx::execution::experimental::schedule_t, my_scheduler_template)
        {
            return {};
        }

        bool operator==(my_scheduler_template const&) const noexcept
        {
            return true;
        }

        bool operator!=(my_scheduler_template const&) const noexcept
        {
            return false;
        }
    };

    using my_scheduler = my_scheduler_template<>;

    struct my_sender
    {
        using is_sender = void;
        template <typename R>
        struct operation_state
        {
            std::decay_t<R> r;
            friend void tag_invoke(hpx::execution::experimental::start_t,
                operation_state& os) noexcept
            {
                hpx::execution::experimental::set_value(std::move(os.r));
            }
        };

        template <typename R>
        friend operation_state<R> tag_invoke(
            hpx::execution::experimental::connect_t, my_sender, R&& r)
        {
            return {std::forward<R>(r)};
        }

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            my_sender const&, Env)
            -> hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t()>;
    };

    // This overload should not be chosen by test_adl_isolation below. We make
    // sure this is a better match than the one in tag_namespace so that if this
    // one is visible it is chosen. It should not be visible.
    template <typename Sender>
    void tag_invoke(tag_namespace::my_tag_t, Sender&&)
    {
        static_assert(sizeof(Sender) == 0);
    }
}    // namespace my_namespace

// This test function expects a type that has my_namespace::my_type as a
// template argument. If template arguments are correctly hidden from ADL the
// friend tag_invoke overload in my_tag_t will be chosen. If template arguments
// are not hidden the unconstrained tag_invoke overload in my_namespace will be
// chosen instead.
template <typename Sender>
void test_adl_isolation(Sender&& sender)
{
    tag_namespace::my_tag(std::forward<Sender>(sender));
}
