//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/meta.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

#if defined(DOXYGEN)
    /// start is a customization point object. The expression
    /// `hpx::execution::experimental::start(r)` is equivalent to:
    ///     * `r.start()`, if that expression is valid. If the function selected
    ///       does not signal the receiver `r`'s done channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `start(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void start();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::functional::tag_invoke`.
    template <typename O>
    void start(O&& o);

    /// An `operation_state` is an object representing the asynchronous operation
    /// that has been returned from calling `hpx::execution::experimental::connect` with
    /// a `sender` and a `receiver`. The only operation on an `operation_state`
    /// is:
    ///     * `hpx::execution::experimental::start`
    ///
    /// `hpx::execution::experimental::start` can be called exactly once. Once it has
    /// been invoked, the caller needs to ensure that the receiver's completion
    /// signaling operations strongly happen before the destructor of the state
    /// is called. The call to `hpx::execution::experimental::start` needs to happen
    /// strongly before the completion signaling operations.
    ///
    template <typename O>
    struct is_operation_state;
#endif

    namespace detail {

        // start should not be callable for operation states that are rvalues
        struct enable_start
        {
            template <typename EnableTag, typename... Ts>
            struct apply : std::false_type
            {
            };

            template <typename EnableTag, typename State>
            struct apply<EnableTag, State> : std::is_lvalue_reference<State>
            {
            };
        };
    }    // namespace detail

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE
    struct start_t
      : hpx::functional::tag_noexcept<start_t, detail::enable_start>
    {
    } start{};

    namespace detail {

        template <bool IsOperationState, typename O>
        struct is_operation_state_impl;

        template <typename O>
        struct is_operation_state_impl<false, O> : std::false_type
        {
        };

        template <typename O>
        struct is_operation_state_impl<true, O>
          : std::integral_constant<bool, noexcept(start(std::declval<O&>()))>
        {
        };
    }    // namespace detail

    template <typename O>
    struct is_operation_state
      : detail::is_operation_state_impl<std::is_destructible_v<O> &&
                std::is_object_v<O> &&
                hpx::is_invocable_v<start_t, std::decay_t<O>&>,
            O>
    {
    };

    template <typename O>
    inline constexpr bool is_operation_state_v =
        meta::value<is_operation_state<O>>;
}    // namespace hpx::execution::experimental
