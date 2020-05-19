//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:hpx::traits::is_callable

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_invoke.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace basic_execution {
#if defined(DOXYGEN)
    /// set_value is a customization point object. The expression
    /// `hpx::basic_execution::set_value(r, as...)` is equivalent to:
    ///     * `r.set_value(as...)`, if that expression is valid. If the function selected
    ///       does not send to value(s) `as...` to the receiver `r`'s value channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_value(r, as...), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void set_value();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::function::tag_invoke`
    template <typename R, typename... As>
    void set_value(R&& r, As&&... as);

    /// set_done is a customization point object. The expression
    /// `hpx::basic_execution::set_done(r)` is equivalent to:
    ///     * `r.set_done()`, if that expression is valid. If the function selected
    ///       does not signal the receiver `r`'s done channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_done(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void set_done();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::function::tag_invoke`
    template <typename R>
    void set_done(R&& r);

    /// set_error is a customization point object. The expression
    /// `hpx::basic_execution::set_error(r, e)` is equivalent to:
    ///     * `r.set_done(e)`, if that expression is valid. If the function selected
    ///       does not send the error e the receiver `r`'s error channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_error(r, e), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void set_error();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::function::tag_invoke`
    template <typename R, typename E>
    void set_error(R&& r, E&& e);
#endif

    /// Receiving values from asynchronous computations is handled by the `receiver`
    /// concept. A `receiver` needs to be able to receive an error or be marked as
    /// being cancelled. As such, the Receiver concept is defined by having the
    /// following two customization points defined, which form the completion-signal
    /// operations:
    ///     * `hpx::basic_execution::set_done`
    ///     * `hpx::basic_execution::set_error`
    ///
    /// Those two functions denote the completion-signal operations. The receiver
    /// contract is as follows:
    ///     * None of a receiver's completion-signal operation shall be invoked
    ///       before `hpx::basic_execution::start` has been called on the operation
    ///       state object that was returned by connecting a receiver to a sender
    ///       `hpx::basic_execution::connect`.
    ///     * Once `hpx::basic_execution::start` has been called on the operation
    ///       state object, exactly one of the receiver's completion-signal operation
    ///       shall complete without an exception before the receiver is destroyed
    ///
    /// Once one of the receiver's completion-signal operation has been completed
    /// without throwing an exception, the receiver contract has been satisfied.
    /// In other words: The asynchronous operation has been completed.
    ///
    /// \see hpx::basic_execution::is_receiver_of
    template <typename T, typename E = std::exception_ptr>
    struct is_receiver;

    /// The `receiver_of` concept is a refinement of the `receiver` concept by
    /// requiring one additional completion-signal operation:
    ///     * `hpx::basic_execution::set_value`
    ///
    /// This completion-signal operation adds the following to the receiver's
    /// contract:
    ///     * If `hpx::basic_execution::set_value` exits with an exception, it
    ///       is still valid to call `hpx::basic_execution::set_error` or
    ///       `hpx::basic_execution::set_done`
    ///
    /// \see hpx::basic_execution::is_receiver
    template <typename T, typename... As>
    struct is_receiver_of;

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_value_t
    {
#define HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION                      \
    hpx::functional::tag_invoke(                                               \
        *this, std::forward<Receiver>(rcv), std::forward<Values>(values)...)
        template <typename Receiver, typename... Values>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, true> /* is tag invocable */,
            Receiver&& rcv, Values&&... values) const
            noexcept(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION))
                -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION                      \
    std::forward<Receiver>(rcv).set_value(std::forward<Values>(values)...)
        template <typename Receiver, typename... Values>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, false> /* is not tag invocable */,
            Receiver&& rcv, Values&&... values) const
            noexcept(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION))
                -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION                      \
    tag_invoke_impl(IsTagInvocable{}, std::forward<Receiver>(rcv),             \
        std::forward<Values>(values)...)
        template <typename Receiver, typename... Values,
            typename IsTagInvocable = hpx::functional::is_tag_invocable<
                set_value_t, Receiver&&, Values&&...>>
        constexpr HPX_FORCEINLINE auto operator()(
            Receiver&& rcv, Values&&... values) const
            noexcept(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION))
                -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_VALUE_EXPRESSION
    } set_value;

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_error_t
    {
#define HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION                      \
    hpx::functional::tag_invoke(                                               \
        *this, std::forward<Receiver>(rcv), std::forward<Error>(error))
        template <typename Receiver, typename Error>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, true> /* is tag invocable */,
            Receiver&& rcv, Error&& error) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION)
        {
            static_assert(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION),
                "hpx::basic_execution::set_error needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION                      \
    std::forward<Receiver>(rcv).set_error(std::forward<Error>(error))
        template <typename Receiver, typename Error>
        constexpr HPX_FORCEINLINE auto tag_invoke_member_impl(
            std::integral_constant<bool, true> /* is noexcept */,
            Receiver&& rcv, Error&& error) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION;
        }
#define HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_IMPL_EXPRESSION                 \
    tag_invoke_member_impl(                                                    \
        std::integral_constant<bool,                                           \
            noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION)>{},    \
        std::forward<Receiver>(rcv), std::forward<Error>(error))
        template <typename Receiver, typename Error>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, false> /* is not tag invocable */,
            Receiver&& rcv, Error&& error) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_IMPL_EXPRESSION)
        {
            static_assert(
                noexcept(
                    HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_IMPL_EXPRESSION),
                "hpx::basic_execution::set_error needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_IMPL_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_IMPL_EXPRESSION
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION                      \
    tag_invoke_impl(IsTagInvocable{}, std::forward<Receiver>(rcv),             \
        std::forward<Error>(error))
        template <typename Receiver, typename Error,
            typename IsTagInvocable = hpx::functional::is_nothrow_tag_invocable<
                set_error_t, Receiver, Error>>
        constexpr HPX_FORCEINLINE auto operator()(
            Receiver&& rcv, Error&& error) const noexcept

            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_ERROR_EXPRESSION
    } set_error;

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_done_t
    {
#define HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION                       \
    hpx::functional::tag_invoke(*this, std::forward<Receiver>(rcv))
        template <typename Receiver>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, true> /* is tag invocable */,
            Receiver&& rcv) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION)
        {
            static_assert(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION),
                "hpx::basic_execution::set_done needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION                       \
    std::forward<Receiver>(rcv).set_done()
        template <typename Receiver>
        constexpr HPX_FORCEINLINE auto tag_invoke_member_impl(
            std::integral_constant<bool, true> /* is not tag invocable */,
            Receiver&& rcv) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION;
        }

#define HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_IMPL_EXPRESSION                  \
    tag_invoke_member_impl(                                                    \
        std::integral_constant<bool,                                           \
            noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION)>{},     \
        std::forward<Receiver>(rcv))
        template <typename Receiver>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::integral_constant<bool, false> /* is not tag invocable */,
            Receiver&& rcv) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_IMPL_EXPRESSION)
        {
            static_assert(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_IMPL_EXPRESSION),
                "hpx::basic_execution::set_done needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_IMPL_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_IMPL_EXPRESSION
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION                       \
    tag_invoke_impl(IsTagInvocable{}, std::forward<Receiver>(rcv))
        template <typename Receiver,
            typename IsTagInvocable = hpx::functional::is_nothrow_tag_invocable<
                hpx::basic_execution::set_done_t, Receiver>>
        constexpr HPX_FORCEINLINE auto operator()(Receiver&& rcv) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_SET_DONE_EXPRESSION
    } set_done;

    namespace detail {
        template <bool ConstructionRequirements, typename T, typename E>
        struct is_receiver_impl;

        template <typename T, typename E>
        struct is_receiver_impl<false, T, E> : std::false_type
        {
        };

        template <typename T, typename E>
        struct is_receiver_impl<true, T, E>
          : std::integral_constant<bool,
                hpx::traits::is_callable<hpx::basic_execution::set_done_t(
                    typename std::decay<T>::type&&)>::value &&
                    hpx::traits::is_callable<hpx::basic_execution::set_error_t(
                        typename std::decay<T>::type&&, E)>::value>
        {
        };
    }    // namespace detail

    template <typename T, typename E>
    struct is_receiver
      : detail::is_receiver_impl<
            std::is_move_constructible<typename std::decay<T>::type>::value &&
                std::is_constructible<typename std::decay<T>::type, T>::value,
            T, E>
    {
    };

    template <typename T, typename E = std::exception_ptr>
    constexpr bool is_receiver_v = is_receiver<T, E>::value;

    namespace detail {
        template <bool IsReceiver, typename T, typename... As>
        struct is_receiver_of_impl;

        template <typename T, typename... As>
        struct is_receiver_of_impl<false, T, As...> : std::false_type
        {
        };

        template <typename T, typename... As>
        struct is_receiver_of_impl<true, T, As...>
          : std::integral_constant<bool,
                hpx::traits::is_callable<hpx::basic_execution::set_value_t(
                    typename std::decay<T>::type&&, As...)>::value>
        {
        };
    }    // namespace detail

    template <typename T, typename... As>
    struct is_receiver_of
      : detail::is_receiver_of_impl<is_receiver_v<T>, T, As...>
    {
    };

    template <typename T, typename... As>
    constexpr bool is_receiver_of_v = is_receiver_of<T, As...>::value;

    namespace detail {
        template <bool IsReceiverOf, typename T, typename... As>
        struct is_nothrow_receiver_of_impl;

        template <typename T, typename... As>
        struct is_nothrow_receiver_of_impl<false, T, As...> : std::false_type
        {
        };

        template <typename T, typename... As>
        struct is_nothrow_receiver_of_impl<true, T, As...>
          : std::integral_constant<bool,
                noexcept(hpx::basic_execution::set_value(
                    std::declval<T>(), std::declval<As>()...))>
        {
        };
    }    // namespace detail

    template <typename T, typename... As>
    struct is_nothrow_receiver_of
      : detail::is_nothrow_receiver_of_impl<is_receiver_v<T>, T, As...>
    {
    };

    template <typename T, typename... As>
    constexpr bool is_nothrow_receiver_of_v =
        is_nothrow_receiver_of<T, As...>::value;
}}    // namespace hpx::basic_execution
