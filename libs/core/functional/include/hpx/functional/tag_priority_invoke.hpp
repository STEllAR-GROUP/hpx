//  Copyright (c) 2020 ETH Zurich
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
        /// The `hpx::functional::tag_override_invoke` name defines a constexpr object
        /// that is invocable with one or more arguments. The first argument
        /// is a 'tag' (typically a CPO). It is only invocable if an overload
        /// of tag_invoke() that accepts the same arguments could be found via
        /// ADL.
        ///
        /// The evaluation of the expression
        /// `hpx::functional::tag_override_invoke(tag, args...)` is
        /// equivalent to evaluating the unqualified call to
        /// `tag_override_invoke(decay-copy(tag), std::forward<Args>(args)...)`.
        ///
        /// `hpx::functional::tag_override_invoke` is implemented against P1895.
        ///
        /// Example:
        /// Defining a new customization point `foo`:
        /// ```
        /// namespace mylib {
        ///     inline constexpr
        ///         struct foo_fn final : hpx::functional::tag_override<foo_fn>
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
        ///     friend constexpr int tag_override_invoke(mylib::foo_fn, bar const& x)
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
        inline constexpr unspecified tag_override_invoke = unspecified;
    }    // namespace unspecified

    /// `hpx::functional::is_tag_override_invocable<Tag, Args...>` is std::true_type if
    /// an overload of `tag_override_invoke(tag, args...)` can be found via ADL.
    template <typename Tag, typename... Args>
    struct is_tag_override_invocable;

    /// `hpx::functional::is_tag_override_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_override_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_tag_override_invocable_v =
        is_tag_override_invocable<Tag, Args...>::value;

    /// `hpx::functional::is_nothrow_tag_override_invocable<Tag, Args...>` is
    /// std::true_type if an overload of `tag_override_invoke(tag, args...)` can be
    /// found via ADL and is noexcept.
    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable;

    /// `hpx::functional::is_tag_override_invocable_v<Tag, Args...>` evaluates to
    /// `hpx::functional::is_tag_override_invocable<Tag, Args...>::value`
    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_override_invocable_v =
        is_nothrow_tag_override_invocable<Tag, Args...>::value;

    /// `hpx::functional::tag_override_invoke_result<Tag, Args...>` is the trait
    /// returning the result type of the call hpx::functioanl::tag_override_invoke. This
    /// can be used in a SFINAE context.
    template <typename Tag, typename... Args>
    using tag_override_invoke_result =
        invoke_result<decltype(tag_override_invoke), Tag, Args...>;

    /// `hpx::functional::tag_override_invoke_result_t<Tag, Args...>` evaluates to
    /// `hpx::functional::tag_override_invoke_result_t<Tag, Args...>::type`
    template <typename Tag, typename... Args>
    using tag_override_invoke_result_t =
        typename tag_override_invoke_result<Tag, Args...>::type;

    /// `hpx::functional::tag_override<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    template <typename Tag>
    struct tag_override;

    /// `hpx::functional::tag_override_noexcept<Tag>` defines a base class that implements
    /// the necessary tag dispatching functionality for a given type `Tag`
    /// where the implementation is required to be noexcept
    template <typename Tag>
    struct tag_override_noexcept;
}}    // namespace hpx::functional
#else

#include <hpx/config.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace functional {

    ///////////////////////////////////////////////////////////////////////////
    namespace tag_override_invoke_t_ns {

        // MSVC needs this, don't ask
        void tag_override_invoke();

        struct tag_override_invoke_t
        {
            template <typename Tag, typename... Ts>
            constexpr HPX_FORCEINLINE auto operator()(Tag tag, Ts&&... ts) const
                noexcept(noexcept(tag_override_invoke(
                    std::declval<Tag>(), std::forward<Ts>(ts)...)))
                    -> decltype(tag_override_invoke(
                        std::declval<Tag>(), std::forward<Ts>(ts)...))
            {
                return tag_override_invoke(tag, std::forward<Ts>(ts)...);
            }

            friend constexpr bool operator==(
                tag_override_invoke_t, tag_override_invoke_t)
            {
                return true;
            }

            friend constexpr bool operator!=(
                tag_override_invoke_t, tag_override_invoke_t)
            {
                return false;
            }
        };
    }    // namespace tag_override_invoke_t_ns

    inline namespace tag_override_invoke_ns {
        HPX_INLINE_CONSTEXPR_VARIABLE
        tag_override_invoke_t_ns::tag_override_invoke_t tag_override_invoke =
            {};
    }    // namespace tag_override_invoke_ns

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag, typename... Args>
    using is_tag_override_invocable =
        hpx::traits::is_invocable<decltype(tag_override_invoke), Tag, Args...>;

    template <typename Tag, typename... Args>
    constexpr bool is_tag_override_invocable_v =
        is_tag_override_invocable<Tag, Args...>::value;

    namespace detail {
        template <typename Sig, bool Invocable>
        struct is_nothrow_tag_override_invocable_impl;

        template <typename Sig>
        struct is_nothrow_tag_override_invocable_impl<Sig, false>
          : std::false_type
        {
        };

        template <typename Tag, typename... Args>
        struct is_nothrow_tag_override_invocable_impl<
            decltype(hpx::functional::tag_override_invoke)(Tag, Args...), true>
          : std::integral_constant<bool,
                noexcept(hpx::functional::tag_override_invoke(
                    std::declval<Tag>(), std::declval<Args>()...))>
        {
        };
    }    // namespace detail

    template <typename Tag, typename... Args>
    struct is_nothrow_tag_override_invocable
      : detail::is_nothrow_tag_override_invocable_impl<
            decltype(hpx::functional::tag_override_invoke)(Tag, Args...),
            is_tag_override_invocable_v<Tag, Args...>>
    {
    };

    template <typename Tag, typename... Args>
    constexpr bool is_nothrow_tag_override_invocable_v =
        is_nothrow_tag_override_invocable<Tag, Args...>::value;

    template <typename Tag, typename... Args>
    using tag_override_invoke_result =
        hpx::util::invoke_result<decltype(hpx::functional::tag_override_invoke),
            Tag, Args...>;

    template <typename Tag, typename... Args>
    using tag_override_invoke_result_t =
        typename tag_override_invoke_result<Tag, Args...>::type;

    ///////////////////////////////////////////////////////////////////////////
    // Helper base class implementing the tag_invoke logic for CPOs that allow
    // overriding user-defined tag_invoke overloads with tag_override_invoke,
    // and that allow setting a fallback with tag_fallback_invoke.
    template <typename Tag>
    struct tag_priority
    {
        // Is tag-override-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                is_tag_override_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const
            noexcept(is_nothrow_tag_override_invocable_v<Tag, Args...>)
                -> tag_override_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_override_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // Is not tag-override-invocable, but tag-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                !is_tag_override_invocable_v<Tag, Args&&...> &&
                is_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const
            noexcept(is_nothrow_tag_invocable_v<Tag, Args...>)
                -> tag_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // Is not tag-override-invocable, not tag-invocable, but
        // tag-fallback-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                !is_tag_override_invocable_v<Tag, Args&&...> &&
                !is_tag_invocable_v<Tag, Args&&...> &&
                is_tag_fallback_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const
            noexcept(is_nothrow_tag_fallback_invocable_v<Tag, Args...>)
                -> tag_fallback_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_fallback_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Helper base class implementing the tag_invoke logic for noexcept CPOs
    // that allow overriding user-defined tag_invoke overloads with
    // tag_override_invoke, and that allow setting a fallback with
    // tag_fallback_invoke.
    template <typename Tag>
    struct tag_priority_noexcept
    {
        // Is nothrow tag-override-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                is_nothrow_tag_override_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const noexcept
            -> tag_override_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_override_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // Is not nothrow tag-override-invocable, but nothrow tag-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                !is_nothrow_tag_override_invocable_v<Tag, Args&&...> &&
                is_nothrow_tag_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const noexcept
            -> tag_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }

        // Is not nothrow tag-override-invocable, not nothrow tag-invocable, but
        // nothrow tag-fallback-invocable
        template <typename... Args,
            typename Enable = typename std::enable_if<
                !is_nothrow_tag_override_invocable_v<Tag, Args&&...> &&
                !is_nothrow_tag_invocable_v<Tag, Args&&...> &&
                is_nothrow_tag_fallback_invocable_v<Tag, Args&&...>>::type>
        constexpr HPX_FORCEINLINE auto operator()(Args&&... args) const noexcept
            -> tag_fallback_invoke_result_t<Tag, Args&&...>
        {
            return hpx::functional::tag_fallback_invoke(
                static_cast<Tag const&>(*this), std::forward<Args>(args)...);
        }
    };
}}    // namespace hpx::functional

#endif
