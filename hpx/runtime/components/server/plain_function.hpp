//  Copyright (c) 2007-2010 Hartmut Kaiser
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
    struct plain_function
    {
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type() 
        { 
            // plain functions are in main memory namespace
            return component_runtime_support; 
        }
        static void set_component_type(component_type) 
        { 
        }
    };

}}}

#endif


