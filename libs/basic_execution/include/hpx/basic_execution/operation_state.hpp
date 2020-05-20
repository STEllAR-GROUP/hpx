//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:hpx::traits::is_callable

#pragma once

#include <hpx/config/constexpr.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_callable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace basic_execution {
#if defined(DOXYGEN)
    /// start is a customization point object. The expression
    /// `hpx::basic_execution::start(r)` is equivalent to:
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
        /// that has been returned from calling `hpx::basic_execution::connect` with
        /// a `sender` and a `receiver`. The only operation on an `operation_state`
        /// is:
        ///     * `hpx::basic_execution::start`
        ///
        /// `hpx::basic_execution::start` can be called exactly once. Once it has
        /// been invoked, the caller needs to ensure that the receiver's completion
        /// signalling operations strongly happen before the destructor of the state
        /// is called. The call to `hpx::basic_exceution::start` needs to happen
        /// strongly before the completion signalling operations.
        ///
        template <typename O>
        struct is_operation_state;
    }    // namespace traits

    HPX_INLINE_CONSTEXPR_VARIABLE struct start_t
    {
#define HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION                          \
    hpx::functional::tag_invoke(*this, std::forward<OperationState>(o))
        template <typename OperationState>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::true_type /* is tag invocable */,
            OperationState&& o) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION)
        {
            static_assert(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION),
                "hpx::basic_execution::start needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION                          \
    std::forward<OperationState>(o).start()
        template <typename OperationState>
        constexpr HPX_FORCEINLINE auto tag_invoke_member_impl(
            std::true_type /* is not tag invocable */,
            OperationState&& o) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION;
        }

#define HPX_BASIC_EXECUTION_RECEIVER_START_IMPL_EXPRESSION                     \
    tag_invoke_member_impl(                                                    \
        std::integral_constant<bool,                                           \
            noexcept(HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION)>{},        \
        std::forward<OperationState>(o))
        template <typename OperationState>
        constexpr HPX_FORCEINLINE auto tag_invoke_impl(
            std::false_type /* is not tag invocable */,
            OperationState&& o) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_START_IMPL_EXPRESSION)
        {
            static_assert(
                noexcept(HPX_BASIC_EXECUTION_RECEIVER_START_IMPL_EXPRESSION),
                "hpx::basic_execution::start needs to be noexcept "
                "invocable");
            return HPX_BASIC_EXECUTION_RECEIVER_START_IMPL_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_START_IMPL_EXPRESSION
#undef HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION

#define HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION                          \
    tag_invoke_impl(IsTagInvocable{}, std::forward<OperationState>(o))
        template <typename OperationState,
            typename IsTagInvocable = hpx::functional::is_nothrow_tag_invocable<
                hpx::basic_execution::start_t, OperationState>>
        constexpr HPX_FORCEINLINE auto operator()(
            OperationState&& o) const noexcept
            -> decltype(HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION)
        {
            return HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION;
        }
#undef HPX_BASIC_EXECUTION_RECEIVER_START_EXPRESSION
    } start;

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
                    noexcept(hpx::basic_execution::start(std::declval<O>()))>
            {
            };
        }    // namespace detail

        template <typename O>
        struct is_operation_state
          : detail::is_operation_state_impl<std::is_destructible<O>::value &&
                    std::is_object<O>::value &&
                    (hpx::traits::is_callable<hpx::basic_execution::start_t(
                            O&)>::value ||
                        hpx::traits::is_callable<hpx::basic_execution::start_t(
                            O const&)>::value ||
                        hpx::traits::is_callable<hpx::basic_execution::start_t(
                            O&&)>::value),
                O>
        {
        };

        template <typename O>
        constexpr bool is_operation_state_v = is_operation_state<O>::value;
    }    // namespace traits
}}       // namespace hpx::basic_execution
