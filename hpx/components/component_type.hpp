//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1057AM)
#define HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1057AM

#include <boost/assert.hpp>

namespace hpx { namespace components
{
    enum component_type
    {
        component_invalid = 0,
        component_factory = 1,      // predefined components needed to create components
        component_px_thread = 2,    // a ParalleX thread

        // test categories
        component_accumulator = 3,  // simple accumulator
        component_memory = 4,       // general memory address
        
        component_graph = 5,        // logical graph (spanning several localities)
        component_local_graph = 6,  // simple graph example component (see examples/graph_component)
        component_graph_vertex = 7, // vertex for simple graph example 
        
        component_last
    };
    
    namespace detail
    {
        char const* const names[] =
        {
            "component_invalid",
            "component_factory",
            "component_px_thread",
            "component_accumulator",
            "component_memory",
            "component_graph",
            "component_local_graph",
            "component_graph_vertex",
        };
    }
        
    inline char const* const get_component_type_name(component_type type)
    {
        if (type >= component_invalid && type < component_last)
            return components::detail::names[type];
        return "<Unknown>";
    }
    
}}

#endif

