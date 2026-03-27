//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_BASE_TRAITS_IS_COMPONENT_HPP
#define HPX_COMPONENTS_BASE_TRAITS_IS_COMPONENT_HPP

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/components_base_fwd.hpp>
#include <cstddef>
#include <type_traits>

namespace hpx::traits {

    template <typename Component, typename Enable>
    struct is_component;

    template <typename Component>
    struct is_fixed_component;

    template <typename Component>
    struct is_managed_component;

    template <typename T, typename Enable>
    struct is_component_or_component_array;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct fixed_component_tag
        {
        };

        struct component_tag
        {
        };

        struct managed_component_tag
        {
        };

        using simple_component_tag = component_tag;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable>
    struct is_component : std::false_type
    {
    };

    template <typename Component>
    struct is_component<Component const, void> : is_component<Component, void>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // Simple components are components
    template <typename Component>
    struct is_component<Component,
        std::enable_if_t<std::is_base_of_v<detail::component_tag, Component>>>
      : std::true_type
    {
    };

    // Fixed components are components
    template <typename Component>
    struct is_component<Component,
        std::enable_if_t<
            std::is_base_of_v<detail::fixed_component_tag, Component>>>
      : std::true_type
    {
    };

    // Managed components are components
    template <typename Component>
    struct is_component<Component,
        std::enable_if_t<
            std::is_base_of_v<detail::managed_component_tag, Component>>>
      : std::true_type
    {
    };

    template <typename Component>
    inline constexpr bool is_component_v = is_component<Component, void>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct is_component_or_component_array : is_component<T, void>
    {
    };

    template <typename T>
    struct is_component_or_component_array<T[], void> : is_component<T, void>
    {
    };

    template <typename T, std::size_t N>
    struct is_component_or_component_array<T[N], void> : is_component<T, void>
    {
    };

    template <typename Component>
    inline constexpr bool is_component_or_component_array_v =
        is_component_or_component_array<Component, void>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    struct is_fixed_component
      : std::is_base_of<traits::detail::fixed_component_tag, Component>
    {
    };

    template <typename Component>
    struct is_fixed_component<Component const> : is_fixed_component<Component>
    {
    };

    template <typename Component>
    inline constexpr bool is_fixed_component_v =
        is_fixed_component<Component>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    struct is_managed_component
      : std::is_base_of<traits::detail::managed_component_tag, Component>
    {
    };

    template <typename Component>
    struct is_managed_component<Component const>
      : is_managed_component<Component>
    {
    };

    template <typename Component>
    inline constexpr bool is_managed_component_v =
        is_managed_component<Component>::value;

}    // namespace hpx::traits

#endif
