//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_MANAGED_COMPONENT_POLICIES_JUN_02_2016_0710PM)
#define HPX_TRAITS_MANAGED_COMPONENT_POLICIES_JUN_02_2016_0710PM

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are constructed
    struct construct_with_back_ptr {};
    struct construct_without_back_ptr {};

    template <typename T, typename Enable = void>
    struct managed_component_ctor_policy
    {
        typedef construct_without_back_ptr type;
    };

    template <typename Component>
    struct managed_component_ctor_policy<Component,
        typename util::always_void<
            typename Component::has_managed_component_base
        >::type>
    {
        typedef typename Component::ctor_policy type;
    };

    ///////////////////////////////////////////////////////////////////////////
    // control the way managed_components are destructed
    struct managed_object_is_lifetime_controlled {};
    struct managed_object_controls_lifetime {};

    template <typename T, typename Enable = void>
    struct managed_component_dtor_policy
    {
        typedef managed_object_controls_lifetime type;
    };

    template <typename Component>
    struct managed_component_dtor_policy<Component,
        typename util::always_void<
            typename Component::has_managed_component_base
        >::type>
    {
        typedef typename Component::dtor_policy type;
    };
}}

#endif
