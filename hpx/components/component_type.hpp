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
        component_factory,          // predefined components needed to create components
        component_px_thread,        // a ParalleX thread

        // LCO's
        component_simple_future,    // a simple future allowing one thread to 
                                    // wait for the result
        
        // test categories
        component_accumulator,      // simple accumulator
        component_memory,           // general memory address
        
        component_graph,            // logical graph (spanning several localities)
        component_local_graph,      // simple graph example component (see examples/graph_component)
        component_graph_vertex,     // vertex for simple graph example 
        
        component_last
    };
    
    namespace detail
    {
        char const* const names[] =
        {
            "component_invalid",
            "component_factory",
            "component_px_thread",
            "component_simple_future",
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

