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

        template <typename T, typename Enable = void>
        struct is_requirable_concept
        {
            static constexpr bool value = false;
        };

        template <typename T>
        struct is_requirable_concept<T,
            std::enable_if_t<T::is_requirable_concept>>
        {
            static constexpr bool value = true;
        };

        template <typename E, typename T>
        struct require_concept_is_applicable
        {
            static constexpr bool value =
                hpx::experimental::is_applicable_property_v<E, T> &&
                is_requirable_concept<std::decay_t<T>>::value;
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct require_concept_t
      : hpx::functional::tag_priority<require_concept_t>
    {
        // This flag enforces the common requirement for the require_concepts
        // CPO that the given property T has to be applicable to the type E and
        // that the property T is marked as is_requirable_concept. The
        // tag_invoke machinery will static_assert if this flag evaluates to
        // false.
        template <typename E, typename T>
        static constexpr bool is_applicable_v =
            detail::require_concept_is_applicable<E, T>::value;

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
            noexcept(noexcept(
                std::declval<E&&>().require_concept(std::forward<T>(t))))
            -> decltype(std::declval<E&&>().require_concept(std::forward<T>(t)))
        {
            return std::forward<E>(e).require_concept(std::forward<T>(t));
        }

        template <typename E, typename T>
        friend constexpr HPX_FORCEINLINE
        auto tag_override_invoke(require_concept_t, E&& e, T&& t)
            noexcept(noexcept(invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t))))
            -> decltype(invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t)))
        {
            return invoke_impl(static_query_value_t<E, T>(),
                std::forward<E>(e), std::forward<T>(t));
        }
        // clang-format on
    } require_concept{};

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    template <typename E, typename T>
    struct can_require_concept
      : std::integral_constant<bool,
            functional::is_tag_priority_invocable_v<require_concept_t, E, T> &&
            detail::require_concept_is_applicable<E, T>::value>
    {
    };

    // clang-format off
    template <typename E, typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool can_require_concept_v =
        can_require_concept<E, T>::value;

    template <typename E, typename T>
    struct is_nothrow_require_concept
      : std::integral_constant<bool,
        functional::is_nothrow_tag_priority_invocable_v<require_concept_t, E, T> &&
        detail::require_concept_is_applicable<E, T>::value>
    {
    };

    template <typename E, typename T>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_nothrow_require_concept_v =
        is_nothrow_require_concept<E, T>::value;

    // clang-format on

}}    // namespace hpx::experimental
