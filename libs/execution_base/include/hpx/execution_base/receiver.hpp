//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution_base {

#if defined(DOXYGEN)
    /// set_value is a customization point object. The expression
    /// `hpx::basic_execution::set_value(r, as...)` is equivalent to:
    ///     * `r.set_value(as...)`, if that expression is valid. If the function selected
    ///       does not send to value(s) `as...` to the Receiver `r`'s value channel,
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
    ///       does not signal the Receiver `r`'s done channel,
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
    ///       does not send the error e the Receiver `r`'s error channel,
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

    namespace traits {

        /// Receiving values from asynchronous computations is handled by the `Receiver`
        /// concept. A `Receiver` needs to be able to receive an error or be marked as
        /// being canceled. As such, the Receiver concept is defined by having the
        /// following two customization points defined, which form the completion-signal
        /// operations:
        ///     * `hpx::basic_execution::set_done`
        ///     * `hpx::basic_execution::set_error`
        ///
        /// Those two functions denote the completion-signal operations. The Receiver
        /// contract is as follows:
        ///     * None of a Receiver's completion-signal operation shall be invoked
        ///       before `hpx::basic_execution::start` has been called on the operation
        ///       state object that was returned by connecting a Receiver to a sender
        ///       `hpx::basic_execution::connect`.
        ///     * Once `hpx::basic_execution::start` has been called on the operation
        ///       state object, exactly one of the Receiver's completion-signal operation
        ///       shall complete without an exception before the Receiver is destroyed
        ///
        /// Once one of the Receiver's completion-signal operation has been completed
        /// without throwing an exception, the Receiver contract has been satisfied.
        /// In other words: The asynchronous operation has been completed.
        ///
        /// \see hpx::basic_execution::traits::is_receiver_of
        template <typename T, typename E = std::exception_ptr>
        struct is_receiver;

        /// The `receiver_of` concept is a refinement of the `Receiver` concept by
        /// requiring one additional completion-signal operation:
        ///     * `hpx::basic_execution::set_value`
        ///
        /// This completion-signal operation adds the following to the Receiver's
        /// contract:
        ///     * If `hpx::basic_execution::set_value` exits with an exception, it
        ///       is still valid to call `hpx::basic_execution::set_error` or
        ///       `hpx::basic_execution::set_done`
        ///
        /// \see hpx::basic_execution::traits::is_receiver
        template <typename T, typename... As>
        struct is_receiver_of;

    }    // namespace traits

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_value_t
      : hpx::functional::tag_fallback<set_value_t>
    {
    private:
        template <typename R, typename... Args>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(set_value_t, R&& r, Args&&... args) noexcept(
            noexcept(
                std::declval<R&&>().set_value(std::forward<Args>(args)...)))
            -> decltype(
                std::declval<R&&>().set_value(std::forward<Args>(args)...))
        {
            return std::forward<R>(r).set_value(std::forward<Args>(args)...);
        }
    } set_value;

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_error_t
      : hpx::functional::tag_fallback_noexcept<set_error_t>
    {
    private:
        template <typename R, typename E>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(set_error_t, R&& r, E&& e) noexcept(
            noexcept(std::declval<R&&>().set_error(std::forward<E>(e))))
            -> decltype(std::declval<R&&>().set_error(std::forward<E>(e)))
        {
            return std::forward<R>(r).set_error(std::forward<E>(e));
        }
    } set_error;

    HPX_INLINE_CONSTEXPR_VARIABLE struct set_done_t
      : hpx::functional::tag_fallback_noexcept<set_done_t>
    {
    private:
        template <typename R>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(set_done_t,
            R&& r) noexcept(noexcept(std::declval<R&&>().set_done()))
            -> decltype(std::declval<R&&>().set_done())
        {
            return std::forward<R>(r).set_done();
        }
    } set_done;

    namespace traits {
        ///////////////////////////////////////////////////////////////////////
        namespace detail {
            template <bool ConstructionRequirements, typename T, typename E>
            struct is_receiver_impl;

            template <typename T, typename E>
            struct is_receiver_impl<false, T, E> : std::false_type
            {
            };

            // clang-format off
            template <typename T, typename E>
            struct is_receiver_impl<true, T, E>
              : std::integral_constant<bool,
                    hpx::traits::is_invocable<
                        hpx::execution_base::set_done_t,
                            typename std::decay<T>::type&&>::value &&
                    hpx::traits::is_invocable<
                        hpx::execution_base::set_error_t,
                            typename std::decay<T>::type&&, E>::value>
            {
            };
            // clang-format on
        }    // namespace detail

        // clang-format off
        template <typename T, typename E>
        struct is_receiver
          : detail::is_receiver_impl<
                std::is_move_constructible<typename std::decay<T>::type>::value &&
                std::is_constructible<typename std::decay<T>::type, T>::value,
                T, E>
        {
        };
        // clang-format on

        template <typename T, typename E = std::exception_ptr>
        constexpr bool is_receiver_v = is_receiver<T, E>::value;

        ///////////////////////////////////////////////////////////////////////
        namespace detail {
            template <bool IsReceiverOf, typename T, typename... As>
            struct is_receiver_of_impl;

            template <typename T, typename... As>
            struct is_receiver_of_impl<false, T, As...> : std::false_type
            {
            };

            // clang-format off
            template <typename T, typename... As>
            struct is_receiver_of_impl<true, T, As...>
              : std::integral_constant<bool,
                    hpx::traits::is_invocable<
                        hpx::execution_base::set_value_t,
                            typename std::decay<T>::type&&, As...>::value>
            {
            };
            // clang-format on
        }    // namespace detail

        template <typename T, typename... As>
        struct is_receiver_of
          : detail::is_receiver_of_impl<is_receiver_v<T>, T, As...>
        {
        };

        template <typename T, typename... As>
        constexpr bool is_receiver_of_v = is_receiver_of<T, As...>::value;

        ///////////////////////////////////////////////////////////////////////
        namespace detail {
            template <bool IsReceiverOf, typename T, typename... As>
            struct is_nothrow_receiver_of_impl;

            template <typename T, typename... As>
            struct is_nothrow_receiver_of_impl<false, T, As...>
              : std::false_type
            {
            };

            template <typename T, typename... As>
            struct is_nothrow_receiver_of_impl<true, T, As...>
              : std::integral_constant<bool,
                    noexcept(hpx::execution_base::set_value(
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

    }    // namespace traits
}}       // namespace hpx::execution_base
