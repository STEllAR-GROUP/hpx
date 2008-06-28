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
        component_runtime_support,  // runtime support (needed to create components, etc.)
        component_memory,           // general memory address
        component_px_thread,        // a ParalleX thread

        // LCO's
        component_base_lco,         // the base of all LCO's not waiting on a value
        component_base_lco_with_value,  // base LCO's blocking on a value
        component_future,           // a future executing the action and allowing to wait for the resul

        component_distributing_factory,   // factory combined with load balancing

        // test categories
        component_accumulator,      // simple accumulator

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
            "component_runtime_support",
            "component_memory",
            "component_px_thread",
            "component_base_lco",
            "component_base_lco_with_value",
            "component_future",
            "component_distributing_factory",
            "component_accumulator",
            "component_graph",
            "component_local_graph",
            "component_graph_vertex",
        };
    }

    ///
    inline char const* const get_component_type_name(int type)
    {
        if (type >= component_invalid && type < component_last)
            return components::detail::names[type];
        return "<Unknown>";
    }

    ///
    inline bool is_valid_component_type(int type)
    {
        return type > component_invalid && type < component_last;
    }

}}

#endif

