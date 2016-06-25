//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_COMPONENT_TYPES_COMPATIBLE_MAR_10_2014_1131AM)
#define HPX_TRAITS_COMPONENT_TYPES_COMPATIBLE_MAR_10_2014_1131AM

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/address.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct component_type_is_compatible
    {
        static bool call(naming::address const& addr)
        {
            return components::types_are_compatible(
                addr.type_, components::get_component_type<Component>());
        }
    };
}}

#endif
