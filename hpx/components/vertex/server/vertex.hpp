//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_VERTEX_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_VERTEX_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The vertex is an HPX component.
    ///
    class vertex
      : public components::detail::managed_component_base<vertex>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the vertex)
        enum actions
        {
            vertex_init = 0,
            vertex_label = 1
        };
        
        // constructor: initialize vertex value
        vertex()
          : label_(-1)
        {}

        ~vertex()
        {
            std::cout << "Dying " << label_ << "\n";
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the vertex
        int init(int label)
        {
            std::cout << "Initializing vertex with label "
                      << label << " on locality"
                      << applier::get_applier().get_runtime_support_gid() << std::endl;

            label_ = label;

            return 0;
        }

        int label(void)
        {
        	return label_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            vertex, int, vertex_init, int, &vertex::init
        > init_action;

        typedef hpx::actions::result_action0<
			vertex, int, vertex_label, &vertex::label
		> label_action;

    private:
        int label_;
    };

}}}

#endif
