//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are constructed
    struct construct_with_back_ptr
    {
    };

    struct construct_without_back_ptr
    {
    };

    template <typename T, typename Enable = void>
    struct managed_component_ctor_policy
    {
        using type = construct_without_back_ptr;
    };

    template <typename Component>
    struct managed_component_ctor_policy<Component,
        typename util::always_void<
            typename Component::has_managed_component_base>::type>
    {
        using type = typename Component::ctor_policy;
    };

    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are destructed
    struct managed_object_is_lifetime_controlled
    {
    };

    struct managed_object_controls_lifetime
    {
    };

    template <typename T, typename Enable = void>
    struct managed_component_dtor_policy
    {
        using type = managed_object_controls_lifetime;
    };

    template <typename Component>
    struct managed_component_dtor_policy<Component,
        typename util::always_void<
            typename Component::has_managed_component_base>::type>
    {
        using type = typename Component::dtor_policy;
    };
}}    // namespace hpx::traits
