//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_PROPS_MAY_17_2008_0731PM)
#define HPX_COMPONENTS_SERVER_PROPS_MAY_17_2008_0731PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

#define LPROPS_(lvl) LAPP_(lvl) << " [PROPS] "

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The props is an HPX component.
    ///
    class props
      : public components::detail::managed_component_base<props>
    {
    private:
        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;

    public:
        // parcel action code: the action to be performed on the destination 
        // object (the props)
        enum actions
        {
            props_init = 0,
            props_color = 1,
            props_incr = 2, // hack so I don't have to make a new comp. for kernel 4
            props_get_score = 3
        };
        
        // constructor: initialize props value
        props()
          : color_(0)
        {
            LPROPS_(info) << "event: action(props::props) status(begin)";
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        int init(int val)
        {
            LPROPS_(info) << "event: action(props::init) status(begin)";
            return val;
        }

        int color(int d)
        {
            mutex_type::scoped_lock l(this);

            LPROPS_(info) << "event: action(props::color) status(begin)";

            if (d > color_)
            {
                color_ = d;
            }

            return color_;
        }

        double incr(double d)
        {
            mutex_type::scoped_lock l(this);

            LPROPS_(info) << "event: action(props::incr) status(begin)";

            count_ += d;

            return count_;
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

        typedef hpx::actions::result_action1<
            props, double, props_incr, double, &props::incr
        > incr_action;

    private:
        int color_;
        double count_;
    };

}}}

#endif
