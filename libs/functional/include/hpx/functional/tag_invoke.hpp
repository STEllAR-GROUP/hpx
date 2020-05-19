//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:hpx::traits::is_callable
// hpxinspect:nodeprecatedname:hpx::util::result_of

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace functional {
    inline namespace unspecified {
        /// The `hpx::functional::tag_invoke` name defines a constexpr opbject that is
        /// invocable with one or more arguments. The first argument is a 'tag' (typically
        /// a CPO). It is only invocable if an overload of tag_invoke() that accepts
        /// the same arguments could be found via ADL.
        ///
        /// The evaluation of the expression `hpx::tag_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_invoke(decay-copy(tag), std::forward<Args>(args)...)`.
        ///
        /// `hpx::functional::tag_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr struct foo_fn {
        ///         template <typename T>
        ///         constexpr auto operator()(T&& x) const
        ///         noexcept(hpx::functional::tag_invoke(*this, std::forward<T>(x)))
        ///             -> decltype(hpx::functional::tag_invoke(*this,  std::forward<T>(x)))
        ///         {
        ///             return hpx::functional::tag_invoke(*this,  std::forward<T>(x));
        ///         }
        ///     } foo{};
        /// }
        /// ```
        ///
        /// Defining an object `bar` which customizes `foo`:
        /// ```
        /// struct bar
        /// {
        ///     int x = 42;
        ///
        ///     friend constexpr int tag_invoke(mylib::foo_fn, bar const& x)
        ///     {
        ///         return b.x;
        ///     }
        /// };
        /// ```
        ///
        /// Using the customization point:
        /// ```
        /// static_assert(42 == mylib::foo(bar{}), "The answer is 42");
        /// ```
        inline constexpr unspecified tag_invoke = unspecified;
    }    // namespace unspecified

    /// `hpx::functional::is_tag_invocable<Tag, Args...>` is std::true_type if an overload
    /// of `tag_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_invocable;

    /// `hpx::functional::is_tag_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_invocable_v = is_tag_invocable<Tag, Args...>::value;

    /// `hpx::functional::is_nothrow_tag_invocable<Tag, Args...>` is std::true_type if an
    /// overload of `tag_invoke(tag, args...)` can be found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_invocable;

    /// `hpx::functional::is_tag_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_invocable_v =
        is_nothrow_tag_invocable<Tag, Args...>::value;

    /// `hpx::functional::tag_invoke_result<Tag, Args...>` is the trait returning the
    /// result type of the call hpx::functioanl::tag_invoke. This can be used in a SFINAE
    /// context.
    template <typename Tag, typename... Args>
    using tag_invoke_result = invoke_result<decltype(tag_invoke), Tag, Args...>;

    /// `hpx::functional::tag_invoke_result_t<Tag, Args...>` evaluates to
    /// `hpx::functional::tag_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_invoke_result_t = typename tag_invoke_result<Tag, Args...>::type;
}}    // namespace hpx::functional
#else

#include <hpx/config.hpp>
#include <hpx/functional/result_of.hpp>
#include <hpx/functional/traits/is_callable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace functional {
    namespace tag_invoke_t_ns {
        struct tag_invoke_t
        {
#define HPX_FUNCTIONAL_TAG_INVOKE_EXPRESSION                                   \
    tag_invoke(tag, std::forward<Args>(args)...) /**/

            template <typename Tag, typename... Args>
            constexpr HPX_FORCEINLINE auto operator()(
                Tag tag, Args&&... args) const
                noexcept(noexcept(HPX_FUNCTIONAL_TAG_INVOKE_EXPRESSION))
                    -> decltype(HPX_FUNCTIONAL_TAG_INVOKE_EXPRESSION)
            {
                return HPX_FUNCTIONAL_TAG_INVOKE_EXPRESSION;
            }
#undef HPX_FUNCTIONAL_TAG_INVOKE_EXPRESSION

            friend constexpr bool operator==(tag_invoke_t, tag_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(tag_invoke_t, tag_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_invoke_t_ns

    inline namespace tag_invoke_ns {
        HPX_INLINE_CONSTEXPR_VARIABLE tag_invoke_t_ns::tag_invoke_t tag_invoke =
            {};
    }

    template <typename Tag, typename... Args>
    using is_tag_invocable =
        hpx::traits::is_callable<decltype(tag_invoke)(Tag, Args...)>;

    template <typename Tag, typename... Args>
    constexpr bool is_tag_invocable_v = is_tag_invocable<Tag, Args...>::value;

    namespace detail {
        template <typename Sig, bool Invocable>
        struct is_nothrow_tag_invocable_impl;

        template <typename Sig>
        struct is_nothrow_tag_invocable_impl<Sig, false> : std::false_type
        {
        };

        template <typename Tag, typename... Args>
        struct is_nothrow_tag_invocable_impl<
            decltype(hpx::functional::tag_invoke)(Tag, Args...), true>
          : std::integral_constant<bool,
                noexcept(hpx::functional::tag_invoke(
                    std::declval<Tag>(), std::declval<Args>()...))>
        {
        };
    }    // namespace detail

    template <typename Tag, typename... Args>
    struct is_nothrow_tag_invocable
      : detail::is_nothrow_tag_invocable_impl<
            decltype(hpx::functional::tag_invoke)(Tag, Args...),
            is_tag_invocable_v<Tag, Args...>>
    {
    };

    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_invocable_v =
        is_nothrow_tag_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_invoke_result = hpx::util::result_of<decltype(
        hpx::functional::tag_invoke)(Tag, Args...)>;

    template <typename Tag, typename... Args>
    using tag_invoke_result_t = typename tag_invoke_result<Tag, Args...>::type;
}}    // namespace hpx::functional

#endif
