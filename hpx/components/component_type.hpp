//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1057AM)
#define HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1057AM

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
        BOOST_ASSERT(type >= component_invalid && type < component_last);
        return components::detail::names[type];
    }
    
}}

///////////////////////////////////////////////////////////////////////////////
// Helper macros for building the execute functions for a component
#define BEGIN_EXECUTE_ACTION(name, action)                                    \
        {                                                                     \
            hpx::components::action_type& __act = action;                     \
            char const* __name = name;                                        \
            switch(__act->get_action_code()) {                                \
    /**/

#define EXECUTE_ACTION(tag, type)                                             \
            case tag:                                                         \
                return boost::static_pointer_cast<type>(__act)                \
                    ->execute(*this);                                         \
    /**/

#define END_EXECUTE_ACTION()                                                  \
            default:                                                          \
                throw hpx::exception(hpx::bad_action_code,                    \
                    std::string("Invalid action code during execute for ") +  \
                        __name + boost::lexical_cast<std::string>(            \
                            __act->get_action_code()));                       \
            }                                                                 \
        }                                                                     \
    /**/

#endif

