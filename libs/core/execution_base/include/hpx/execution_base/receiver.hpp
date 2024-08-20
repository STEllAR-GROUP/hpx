//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

#if defined(DOXYGEN)
    /// set_value is a customization point object. The expression
    /// `hpx::execution::set_value(r, as...)` is equivalent to:
    ///     * `r.set_value(as...)`, if that expression is valid. If the function
    ///       selected does not send the value(s) `as...` to the Receiver `r`'s
    ///       value channel, the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_value(r, as...), if that expression is valid, with
    ///       overload resolution performed in a context that include the
    ///       declaration `void set_value();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename R, typename... As>
    void set_value(R&& r, As&&... as);

    /// set_stopped is a customization point object. The expression
    /// `hpx::execution::set_stopped(r)` is equivalent to:
    ///     * `r.set_stopped()`, if that expression is valid. If the function
    ///       selected does not signal the Receiver `r`'s done channel, the
    ///       program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_stopped(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the
    ///       declaration `void set_stopped();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename R>
    void set_stopped(R&& r);

    /// set_error is a customization point object. The expression
    /// `hpx::execution::set_error(r, e)` is equivalent to:
    ///     * `r.set_stopped(e)`, if that expression is valid. If the function
    ///       selected does not send the error `e` the Receiver `r`'s error
    ///       channel, the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `set_error(r, e), if that expression is valid, with
    ///       overload resolution performed in a context that include the
    ///       declaration `void set_error();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of
    /// `hpx::functional::tag_invoke`.
    template <typename R, typename E>
    void set_error(R&& r, E&& e);
#endif

    /// Receiving values from asynchronous computations is handled by the
    /// `Receiver` concept. A `Receiver` needs to be able to receive an error or
    /// be marked as being canceled. As such, the Receiver concept is defined by
    /// having the following two customization points defined, which form the
    /// completion-signal operations:
    ///     * `hpx::execution::experimental::set_stopped` *
    ///       `hpx::execution::experimental::set_error`
    ///
    /// Those two functions denote the completion-signal operations. The
    /// Receiver contract is as follows:
    ///     * None of a Receiver's completion-signal operation shall be invoked
    ///       before `hpx::execution::experimental::start` has been called on
    ///       the operation state object that was returned by connecting a
    ///       Receiver to a sender `hpx::execution::experimental::connect`.
    ///     * Once `hpx::execution::start` has been called on the operation
    ///       state object, exactly one of the Receiver's completion-signal
    ///       operation shall complete without an exception before the Receiver
    ///       is destroyed
    ///
    /// Once one of the Receiver's completion-signal operation has been
    /// completed without throwing an exception, the Receiver contract has been
    /// satisfied. In other words: The asynchronous operation has been
    /// completed.
    ///
    /// \see hpx::execution::experimental::is_receiver_of
    template <typename T, typename E = std::exception_ptr>
    struct is_receiver;

    /// The `receiver_of` concept is a refinement of the `Receiver` concept by
    /// requiring one additional completion-signal operation:
    ///     * `hpx::execution::set_value`
    ///
    /// The `receiver_of` concept takes a receiver and an instance of the
    /// `completion_signatures<>` class template. The `receiver_of` concept,
    /// rather than accepting a receiver and some value types, is changed to
    /// take a receiver and an instance of the `completion_signatures<>` class
    /// template. A sender uses `completion_signatures<>` to describe the
    /// signals with which it completes. The `receiver_of` concept ensures that
    /// a particular receiver is capable of receiving those signals.
    ///
    /// This completion-signal operation adds the following to the Receiver's
    /// contract:
    ///     * If `hpx::execution::set_value` exits with an exception, it
    ///       is still valid to call `hpx::execution::set_error` or
    ///       `hpx::execution::set_stopped`
    ///
    /// \see hpx::execution::traits::is_receiver
    template <typename Receiver, typename CS>
    struct is_receiver_of;

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_value_t : hpx::functional::tag<set_value_t>
    {
    } set_value{};

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_error_t : hpx::functional::tag_noexcept<set_error_t>
    {
    } set_error{};

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct set_stopped_t : hpx::functional::tag_noexcept<set_stopped_t>
    {
    } set_stopped{};

    ///////////////////////////////////////////////////////////////////////
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
                hpx::is_invocable_v<set_stopped_t, std::decay_t<T>&&> &&
                    hpx::is_invocable_v<set_error_t, std::decay_t<T>&&, E>>
        {
        };
    }    // namespace detail

    template <typename T, typename E>
    struct is_receiver
      : detail::is_receiver_impl<
            std::is_move_constructible_v<std::decay_t<T>> &&
                std::is_constructible_v<std::decay_t<T>, T>,
            T, E>
    {
    };

    template <typename T, typename E = std::exception_ptr>
    inline constexpr bool is_receiver_v = is_receiver<T, E>::value;

    ///////////////////////////////////////////////////////////////////////
    namespace detail {

        template <bool IsReceiverOf, typename T, typename CS>
        struct is_receiver_of_impl;

        template <typename T, typename CS>
        struct is_receiver_of_impl<false, T, CS> : std::false_type
        {
        };

        template <typename F, typename T, typename Variant>
        struct is_invocable_variant_of_tuples : std::false_type
        {
        };

        template <typename F, typename T, typename... Ts>
        struct is_invocable_variant_of_tuples<F, T,
            meta::pack<meta::pack<Ts...>>>
          : hpx::is_invocable<F, std::decay_t<T>&&, Ts...>
        {
        };

        template <typename F, typename T, typename Variant>
        struct is_invocable_variant : std::false_type
        {
        };

        template <typename F, typename T, typename... Ts>
        struct is_invocable_variant<F, T, meta::pack<Ts...>>
          : hpx::is_invocable<F, std::decay_t<T>&&, Ts...>
        {
        };

        template <typename T, typename CS>
        struct is_receiver_of_impl<true, T, CS>
          : std::integral_constant<bool,
                is_invocable_variant_of_tuples<set_value_t, T,
                    typename CS::template value_types<meta::pack,
                        meta::pack>>::value &&
                    is_invocable_variant<set_error_t, T,
                        typename CS::template error_types<meta::pack>>::value &&
                    CS::sends_stopped>
        {
        };
    }    // namespace detail

    template <typename T, typename CS>
    struct is_receiver_of : detail::is_receiver_of_impl<is_receiver_v<T>, T, CS>
    {
    };

    template <typename T, typename CS>
    inline constexpr bool is_receiver_of_v = is_receiver_of<T, CS>::value;

    ///////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename F, typename T, typename Variant>
        struct is_nothrow_invocable_variant_of_tuples : std::false_type
        {
        };

        template <typename F, typename T, typename... Ts>
        struct is_nothrow_invocable_variant_of_tuples<F, T,
            meta::pack<meta::pack<Ts...>>>
          : hpx::functional::is_nothrow_tag_invocable<F, std::decay_t<T>&&,
                Ts...>
        {
        };

        template <typename F, typename T, typename Variant>
        struct is_nothrow_invocable_variant : std::false_type
        {
        };

        template <typename F, typename T, typename... Ts>
        struct is_nothrow_invocable_variant<F, T, meta::pack<Ts...>>
          : hpx::functional::is_nothrow_tag_invocable<F, std::decay_t<T>&&,
                Ts...>
        {
        };

        template <bool IsReceiverOf, typename T, typename CS>
        struct is_nothrow_receiver_of_impl;

        template <typename T, typename CS>
        struct is_nothrow_receiver_of_impl<false, T, CS> : std::false_type
        {
        };

        template <typename T, typename CS>
        struct is_nothrow_receiver_of_impl<true, T, CS>
          : std::integral_constant<bool,
                is_nothrow_invocable_variant_of_tuples<set_value_t, T,
                    typename CS::template value_types<meta::pack,
                        meta::pack>>::value &&
                    is_nothrow_invocable_variant<set_error_t, T,
                        typename CS::template error_types<meta::pack>>::value &&
                    CS::sends_stopped>
        {
        };
    }    // namespace detail

    template <typename T, typename CS>
    struct is_nothrow_receiver_of
      : detail::is_nothrow_receiver_of_impl<
            is_receiver_v<T> && is_receiver_of_v<T, CS>, T, CS>
    {
    };

    template <typename T, typename CS>
    inline constexpr bool is_nothrow_receiver_of_v =
        is_nothrow_receiver_of<T, CS>::value;

    namespace detail {

        template <typename CPO>
        struct is_receiver_cpo : std::false_type
        {
        };

        template <>
        struct is_receiver_cpo<set_value_t> : std::true_type
        {
        };

        template <>
        struct is_receiver_cpo<set_error_t> : std::true_type
        {
        };

        template <>
        struct is_receiver_cpo<set_stopped_t> : std::true_type
        {
        };

        template <typename CPO>
        inline constexpr bool is_receiver_cpo_v = is_receiver_cpo<CPO>::value;
    }    // namespace detail
}    // namespace hpx::execution::experimental
