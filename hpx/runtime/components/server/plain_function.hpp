//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_COMPONENTS_SERVER_PLAIN_COMPONENT_NOV_14_2008_0726PM)
#define HPX_RUNTIME_COMPONENTS_SERVER_PLAIN_COMPONENT_NOV_14_2008_0726PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    /// placeholder type allowing to integrate the plain action templates
    /// with the existing component based action template infrastructure
    template <typename Action>
    struct plain_function
    {
        static component_type get_component_type()
        {
            return components::get_component_type<plain_function<Action> >();
        }
        static void set_component_type(component_type type)
        {
            components::set_component_type<plain_function<Action> >(type);
        }
    };
}}}

#endif


