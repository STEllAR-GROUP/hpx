// Copyright (c) 2021 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/tag_priority_invoke.hpp>
#include <hpx/properties/is_applicable_property.hpp>
#include <hpx/properties/static_query.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace experimental {

    namespace detail {

        template <typename E, typename T>
        struct query_is_applicable
        {
            static constexpr bool value =
                hpx::experimental::is_applicable_property_v<E, T>;
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct query_t
      : hpx::functional::tag_priority<query_t>
    {
        // This flag enforces the common requirement for the query CPO that the
        // given property T has to be applicable to the type E. The tag_invoke
        // machinery will static_assert if this flag evaluates to false.
        template <typename E, typename T>
        static constexpr bool is_applicable_v =
            detail::query_is_applicable<E, T>::value;

    private:
        // clang-format off
        template <typename E, typename T>
        static constexpr HPX_FORCEINLINE
        auto invoke_impl(std::true_type, E&&, T&&)
            noexcept(noexcept(static_query<E, T>::property_value()))
            -> decltype(static_query<E, T>::property_value())
        {
            return static_query<E, T>::property_value();
        }

        template <typename E, typename T>
        static constexpr HPX_FORCEINLINE
        auto invoke_impl(std::false_type, E&& e, T&& t)
            noexcept(noexcept(
                std::declval<E&&>().query(std::forward<T>(t))))
            -> decltype(std::declval<E&&>().query(std::forward<T>(t)))
        {
            return std::forward<E>(e).query(std::forward<T>(t));
        }

        template <typename E, typename T>
        friend constexpr HPX_FORCEINLINE
        auto tag_override_invoke(query_t, E&& e, T&& t)
            noexcept(noexcept(invoke_impl(static_query_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t))))
            -> decltype(invoke_impl(static_query_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t)))
        {
            return invoke_impl(static_query_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t));
        }
        // clang-format on
    } query{};

    ///////////////////////////////////////////////////////////////////////////
    template <typename E, typename T>
    struct can_query
      : std::integral_constant<bool,
            functional::is_tag_priority_invocable_v<query_t, E, T> &&
                detail::query_is_applicable<E, T>::value>
    {
    };

    template <typename E, typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_query_v = can_query<E, T>::value;

    template <typename E, typename T>
    struct is_nothrow_query
      : std::integral_constant<bool,
            functional::is_nothrow_tag_priority_invocable_v<query_t, E, T> &&
                detail::query_is_applicable<E, T>::value>
    {
    };

    template <typename E, typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_nothrow_query_v =
        is_nothrow_query<E, T>::value;

}}    // namespace hpx::experimental
