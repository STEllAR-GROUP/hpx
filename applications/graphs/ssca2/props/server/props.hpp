//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_PROPS_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_PROPS_MAY_17_2008_0731PM

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
    /// The props is an HPX component.
    ///
    class props
      : public components::detail::managed_component_base<props>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the props)
        enum actions
        {
            props_init = 0,
            props_color = 1
        };
        
        // constructor: initialize props value
        props()
          : color_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int init(int val)
        {
            return val;
        }

        int color(int d)
        {
            if (d > color_)
            {
                color_ = d-1;
            }

            return color_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            props, int, props_init, int, &props::init
        > init_action;

        typedef hpx::actions::result_action1<
            props, int, props_color, int, &props::color
        > color_action;

    private:
        int color_;
    };

}}}

#endif
