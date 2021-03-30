// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// 'require' is a dangerous Apple macro, silence inspect about this
// hpxinspect:noapple_macros:require

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/tag_priority_invoke.hpp>
#include <hpx/properties/is_applicable_property.hpp>
#include <hpx/properties/static_query.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace experimental {

    namespace detail {

        template <typename E, typename... T>
        struct require_is_applicable;

        template <typename T, typename Enable = void>
        struct is_requirable
        {
            static constexpr bool value = false;
        };

        template <typename T>
        struct is_requirable<T, std::enable_if_t<T::is_requirable>>
        {
            static constexpr bool value = true;
        };

        template <typename E, typename T>
        struct require_is_applicable<E, T>
        {
            static constexpr bool value =
                hpx::experimental::is_applicable_property_v<E, T> &&
                is_requirable<std::decay_t<T>>::value;
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct require_t
      : hpx::functional::tag_priority<require_t>
    {
        // This flag enforces the common requirement for the requires CPO that
        // the given property T has to be applicable to the type E and that the
        // property T is marked as requirable. The tag_invoke machinery will
        // static_assert if this flag evaluates to false.
        template <typename E, typename... Tn>
        static constexpr bool is_applicable_v =
            detail::require_is_applicable<E, Tn...>::value;

    private:
        // clang-format off
        template <typename E, typename T>
        static constexpr HPX_FORCEINLINE
        auto invoke_impl(std::true_type, E&& e, T&&)
            noexcept(noexcept(std::forward<E>(e)))
            -> decltype(std::forward<E>(e))
        {
            return std::forward<E>(e);
        }

        template <typename E, typename T>
        static constexpr HPX_FORCEINLINE
        auto invoke_impl(std::false_type, E&& e, T&& t)
            noexcept(noexcept(std::declval<E&&>().require(std::forward<T>(t))))
            -> decltype(std::declval<E&&>().require(std::forward<T>(t)))
        {
            return std::forward<E>(e).require(std::forward<T>(t));
        }

        template <typename E, typename T>
        friend constexpr HPX_FORCEINLINE
        auto tag_override_invoke(require_t, E&& e, T&& t)
            noexcept(noexcept(invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t))))
            -> decltype(invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t)))
        {
            return invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t));
        }
        // clang-format on
    } require{};

    ///////////////////////////////////////////////////////////////////////////
    // Specialization for the case that require was invoked with at least two
    // properties.

    // clang-format off
    namespace detail {

        template <typename E, typename T0, typename T1, typename Pack,
            typename Enable = void>
        struct require_is_applicable_helper
        {
            static constexpr bool value = false;
        };

        template <typename E, typename T0, typename T1, typename... Tn>
        struct require_is_applicable_helper<E, T0, T1, util::pack<Tn...>,
            std::enable_if_t<
                functional::is_tag_priority_invocable_v<require_t, E, T0> &&
                require_is_applicable<E, T0>::value>>
        {
            static constexpr bool value =
                require_is_applicable<decltype(require(
                                  std::declval<E&&>(), std::declval<T0&&>())),
                    T1, Tn...>::value;
        };

        template <typename E, typename T0, typename T1, typename... Tn>
        struct require_is_applicable<E, T0, T1, Tn...>
        {
            static constexpr bool value = require_is_applicable_helper<
                E, T0, T1, util::pack<Tn...>>::value;
        };
    }    // namespace detail

    template <typename E, typename T0, typename T1, typename... Tn>
    constexpr HPX_FORCEINLINE
    auto tag_fallback_invoke(require_t, E&& e, T0&& t0, T1&& t1, Tn&&... tn)
        noexcept(noexcept(
            require(
                require(std::forward<E>(e), std::forward<T0>(t0)),
                std::forward<T1>(t1), std::forward<Tn>(tn)...)
        ))
        -> decltype(require(
                require(std::forward<E>(e), std::forward<T0>(t0)),
                std::forward<T1>(t1), std::forward<Tn>(tn)...))
    {
        return require(
            require(std::forward<E>(e), std::forward<T0>(t0)),
            std::forward<T1>(t1), std::forward<Tn>(tn)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename... Tn>
    struct can_require
      : std::integral_constant<bool,
            functional::is_tag_priority_invocable_v<require_t, T, Tn...> &&
            detail::require_is_applicable<T, Tn...>::value>
    {
    };

    template <typename T, typename... Tn>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_require_v =
        can_require<T, Tn...>::value;

    template <typename T, typename... Tn>
    struct is_nothrow_require
      : std::integral_constant<bool,
            functional::is_nothrow_tag_priority_invocable_v<require_t, T, Tn...> &&
            detail::require_is_applicable<T, Tn...>::value>
    {
    };

    template <typename T, typename... Tn>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_nothrow_require_v =
        is_nothrow_require<T, Tn...>::value;

    // clang-format on

}}    // namespace hpx::experimental
