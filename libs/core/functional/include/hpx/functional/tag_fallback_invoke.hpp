//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace functional {
    inline namespace unspecified {
        /// The `hpx::functional::tag_fallback_invoke` name defines a constexpr object
        /// that is invocable with one or more arguments. The first argument
        /// is a 'tag' (typically a CPO). It is only invocable if an overload
        /// of tag_fallback_invoke() that accepts the same arguments could be
        /// found via ADL.
        ///
        /// The evaluation of the expression
        /// `hpx::functional::tag_fallback_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_fallback_invoke(decay-copy(tag), std::forward<Args>(args)...)`.
        ///
        /// `hpx::functional::tag_fallback_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr
        ///         struct foo_fn final : hpx::functional::tag_fallback<foo_fn>
        ///         {
        ///         } foo{};
        /// }
        /// ```
        ///
        /// Defining an object `bar` which customizes `foo`:
        /// ```
        /// struct bar
        /// {
        ///     int x = 42;
        ///
        ///     friend constexpr int tag_fallback_invoke(mylib::foo_fn, bar const& x)
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
        inline constexpr unspecified tag_fallback_invoke = unspecified;
    }    // namespace unspecified

    /// `hpx::functional::is_tag_fallback_invocable<Tag, Args...>` is std::true_type if
    /// an overload of `tag_fallback_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_fallback_invocable;

    /// `hpx::functional::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_fallback_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_fallback_invocable_v =
        is_tag_fallback_invocable<Tag, Args...>::value;

    /// `hpx::functional::is_nothrow_tag_fallback_invocable<Tag, Args...>` is
    /// std::true_type if an overload of `tag_fallback_invoke(tag, args...)` can be
    /// found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_fallback_invocable;

    /// `hpx::functional::is_tag_fallback_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_fallback_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_fallback_invocable_v =
        is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

    /// `hpx::functional::tag_fallback_invoke_result<Tag, Args...>` is the trait
    /// returning the result type of the call hpx::functioanl::tag_fallback_invoke. This
    /// can be used in a SFINAE context.
    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result =
        invoke_result<decltype(tag_fallback_invoke), Tag, Args...>;

    /// `hpx::functional::tag_fallback_invoke_result_t<Tag, Args...>` evaluates to
    /// `hpx::functional::tag_fallback_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result_t =
        typename tag_fallback_invoke_result<Tag, Args...>::type;

    /// `hpx::functional::tag_fallback<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    template <typename Tag>
    struct tag_fallback;

    /// `hpx::functional::tag_fallback_noexcept<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    /// where the implementation is required to be noexcept
    template <typename Tag>
    struct tag_fallback_noexcept;
}}    // namespace hpx::functional
#else

#include <hpx/config.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/tag_invoke_is_applicable.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace functional {

    ///////////////////////////////////////////////////////////////////////////
    namespace tag_fallback_invoke_t_ns {

        // MSVC needs this, don't ask
        void tag_fallback_invoke();

        struct tag_fallback_invoke_t
        {
            template <typename Tag, typename... Ts>
            constexpr HPX_FORCEINLINE auto operator()(Tag tag, Ts&&... ts) const
                noexcept(noexcept(tag_fallback_invoke(
                    std::declval<Tag>(), std::forward<Ts>(ts)...)))
                    -> decltype(tag_fallback_invoke(
                        std::declval<Tag>(), std::forward<Ts>(ts)...))
            {
                return tag_fallback_invoke(tag, std::forward<Ts>(ts)...);
            }

            friend constexpr bool operator==(
                tag_fallback_invoke_t, tag_fallback_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(
                tag_fallback_invoke_t, tag_fallback_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_fallback_invoke_t_ns

    inline namespace tag_fallback_invoke_ns {
        HPX_INLINE_CONSTEXPR_VARIABLE
        tag_fallback_invoke_t_ns::tag_fallback_invoke_t tag_fallback_invoke =
            {};
    }    // namespace tag_fallback_invoke_ns

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag, typename... Args>
    using is_tag_fallback_invocable =
        hpx::is_invocable<decltype(tag_fallback_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    constexpr bool is_tag_fallback_invocable_v =
        is_tag_fallback_invocable<Tag, Args...>::value;

    namespace detail {
        template <typename Sig, bool Invocable>
        struct is_nothrow_tag_fallback_invocable_impl;

        template <typename Sig>
        struct is_nothrow_tag_fallback_invocable_impl<Sig, false>
          : std::false_type
        {
        };

        template <typename Tag, typename... Args>
        struct is_nothrow_tag_fallback_invocable_impl<
            decltype(hpx::functional::tag_fallback_invoke)(Tag, Args...), true>
          : std::integral_constant<bool,
                noexcept(hpx::functional::tag_fallback_invoke(
                    std::declval<Tag>(), std::declval<Args>()...))>
        {
        };
    }    // namespace detail

    template <typename Tag, typename... Args>
    struct is_nothrow_tag_fallback_invocable
      : detail::is_nothrow_tag_fallback_invocable_impl<
            decltype(hpx::functional::tag_fallback_invoke)(Tag, Args...),
            is_tag_fallback_invocable_v<Tag, Args...>>
    {
    };

    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_fallback_invocable_v =
        is_nothrow_tag_fallback_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result =
        hpx::util::invoke_result<decltype(hpx::functional::tag_fallback_invoke),
            Tag, Args...>;

    template <typename Tag, typename... Args>
    using tag_fallback_invoke_result_t =
        typename tag_fallback_invoke_result<Tag, Args...>::type;

    ///////////////////////////////////////////////////////////////////////////
    /// Helper base class implementing the tag_invoke logic for CPOs that fall
    /// back to directly invoke its fallback.
    ///
    /// This base class is in many cases preferable to the plain tag base class.
    /// With the normal tag base class a default, unconstrained, default
    /// tag_invoke overload will take precedence over user-defined tag_invoke
    /// overloads that are not perfect matches. For example, with a default
    /// overload:
    ///
    /// template <typename T> auto tag_invoke(tag_t, T&& t) {...}
    ///
    /// and a user-defined overload in another namespace:
    ///
    /// auto tag_invoke(my_type t)
    ///
    /// the user-defined overload will only be considered when it is an exact
    /// match. This means const and reference qualifiers must match exactly, and
    /// conversions to a base class are not considered.
    ///
    /// With tag_fallback one can define the default implementation in terms of
    /// a tag_fallback_invoke overload instead of tag_invoke:
    ///
    /// template <typename T> auto tag_fallback_invoke(tag_t, T&& t) {...}
    ///
    /// With the same user-defined tag_invoke overload, the user-defined
    /// overload will now be used if is a match even if isn't an exact match.
    /// This is because tag_fallback will dispatch to tag_fallback_invoke only
    /// if there are no matching tag_invoke overloads.
    template <typename Tag>
    struct tag_fallback
    {
        // is tag-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                is_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const
            noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
                -> tag_invoke_result_t<Tag, Args&&...>
        {
            static_assert(
                hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>,
                "hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>");

            return hpx::functional::tag_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // is not tag-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                !is_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const
            noexcept(is_nothrow_tag_fallback_invocable_v<Tag, Args...>)
                -> tag_fallback_invoke_result_t<Tag, Args&&...>
        {
            static_assert(
                hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>,
                "hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>");

            return hpx::functional::tag_fallback_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // helper base class implementing the tag_invoke logic for CPOs that fall
    // back to directly invoke its fallback. Either invocation has to be noexcept.
    template <typename Tag>
    struct tag_fallback_noexcept
    {
    private:
        // is nothrow tag-fallback invocable
        template <typename... Args>
        constexpr HPX_FORCEINLINE auto tag_fallback_invoke_impl(
            std::true_type, Args&&... args) const noexcept
            -> tag_fallback_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_fallback_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

    public:
        // is nothrow tag-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                is_nothrow_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const noexcept
            -> tag_invoke_result_t<Tag, Args&&...>
        {
            static_assert(
                hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>,
                "hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>");

            return hpx::functional::tag_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // is not nothrow tag-invocable
        template <typename... Args,
            typename IsFallbackInvocable =
                is_nothrow_tag_fallback_invocable<Tag, Args&&...>,
            typename Enable = typename std::enable_if<
                !is_nothrow_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const noexcept
            -> decltype(tag_fallback_invoke_impl(
                IsFallbackInvocable{}, std::forward<Args>(args)...))
        {
            static_assert(
                hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>,
                "hpx::functional::is_tag_invoke_applicable_v<Tag, Args...>");

            return tag_fallback_invoke_impl(
                IsFallbackInvocable{}, std::forward<Args>(args)...);
        }
    };
}}    // namespace hpx::functional

#endif
