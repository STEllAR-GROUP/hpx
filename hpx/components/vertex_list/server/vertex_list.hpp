//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_VERTEX_LIST_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_VERTEX_LIST_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The vertex_list is an HPX component. 
    ///
    class HPX_COMPONENT_EXPORT vertex_list 
      : public simple_component_base<vertex_list> 
    {
    private:
        typedef simple_component_base<vertex_list> base_type;
        
    public:
        vertex_list();
        
        //typedef vertex_list::server::vertex_list wrapping_type;
        typedef hpx::components::server::vertex_list wrapping_type;
        
        // parcel action code: the action to be performed on the destination 
        // object (the vertex_list)
        enum actions
        {
            vertex_list_init = 0
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the vertex_list
        int init(components::component_type item_type, std::size_t num_items); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            vertex_list, int, vertex_list_init, components::component_type, std::size_t, &vertex_list::init
        > init_action;
        
    protected:
        typedef components::distributing_factory::iterator_range_type
            distributed_iterator_range_type;
            
    private:
        //typedef components::distributing_factory::result_type result_type;
//        std::vector<result_type> blocks_;
        std::size_t num_items_;
        std::vector<naming::id_type> blocks_;
    };

}}}

#endif
