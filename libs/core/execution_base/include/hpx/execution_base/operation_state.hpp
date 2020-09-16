//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace execution_base { namespace experimental {
#if defined(DOXYGEN)
    /// start is a customization point object. The expression
    /// `hpx::execution_base::experimental::start(r)` is equivalent to:
    ///     * `r.start()`, if that expression is valid. If the function selected
    ///       does not signal the receiver `r`'s done channel,
    ///       the program is ill-formed (no diagnostic required).
    ///     * Otherwise, `start(r), if that expression is valid, with
    ///       overload resolution performed in a context that include the declaration
    ///       `void start();`
    ///     * Otherwise, the expression is ill-formed.
    ///
    /// The customization is implemented in terms of `hpx::function::tag_invoke`
    template <typename O>
    void start(O&& o);
#endif

    namespace traits {
        /// An `operation_state` is an object representing the asynchronous operation
        /// that has been returned from calling `hpx::execution_base::experimental::connect` with
        /// a `sender` and a `receiver`. The only operation on an `operation_state`
        /// is:
        ///     * `hpx::execution_base::experimental::start`
        ///
        /// `hpx::execution_base::experimental::start` can be called exactly once. Once it has
        /// been invoked, the caller needs to ensure that the receiver's completion
        /// signaling operations strongly happen before the destructor of the state
        /// is called. The call to `hpx::basic_exceution::start` needs to happen
        /// strongly before the completion signaling operations.
        ///
        template <typename O>
        struct is_operation_state;
    }    // namespace traits

    HPX_INLINE_CONSTEXPR_VARIABLE struct start_t
      : hpx::functional::tag_fallback_noexcept<start_t>
    {
    private:
        template <typename OperationState>
        friend constexpr HPX_FORCEINLINE auto
        tag_fallback_invoke(start_t, OperationState&& o) noexcept(
            noexcept(std::declval<OperationState&&>().start()))
            -> decltype(std::declval<OperationState&&>().start())
        {
            return std::forward<OperationState>(o).start();
        }
    } start{};

    namespace traits {
        namespace detail {
            template <bool IsOperationState, typename O>
            struct is_operation_state_impl;

            template <typename O>
            struct is_operation_state_impl<false, O> : std::false_type
            {
            };

            template <typename O>
            struct is_operation_state_impl<true, O>
              : std::integral_constant<bool,
                    noexcept(hpx::execution_base::experimental::start(
                        std::declval<O&&>()))>
            {
            };
        }    // namespace detail

        template <typename O>
        struct is_operation_state
          : detail::is_operation_state_impl<std::is_destructible<O>::value &&
                    std::is_object<O>::value &&
                    hpx::traits::is_invocable<
                        hpx::execution_base::experimental::start_t,
                        typename std::decay<O>::type&&>::value,
                O>
        {
        };

        template <typename O>
        constexpr bool is_operation_state_v = is_operation_state<O>::value;
    }    // namespace traits
}}}      // namespace hpx::execution_base::experimental
