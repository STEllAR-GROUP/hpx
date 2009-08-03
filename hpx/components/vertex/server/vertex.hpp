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
            vertex_init = 0
        };
        
        // constructor: initialize vertex value
        vertex()
          : label_(-1)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the vertex
        int init(int label)
        {            
            if (label_ == -1)
            {
                std::cout << "Setting label '" << label << "' "
                          << "on locality '" << applier::get_applier().get_runtime_support_gid() << "'"
                          << "\n";
                label_ = label;
            }
            else
            {
                std::cout << "Resetting label from '" << label_ << "' to '" << label << "' "
                           << "on locality '" << applier::get_applier().get_runtime_support_gid() << "'"
                           << "\n";
                label_ = label;
            }

            return 0;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            vertex, int, vertex_init, int, &vertex::init
        > init_action;

    private:
        int label_;
    };

}}}

#endif
