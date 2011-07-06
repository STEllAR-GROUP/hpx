//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_point)
#define HPX_COMPONENTS_SERVER_point

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server 
{
    class point
      : public components::detail::managed_component_base<point> 
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            point_init = 0,
            point_get_X = 1,
            point_get_Y = 2,
            point_set_X = 3,
            point_set_Y = 4
        };

        // constructor: initialize accumulator value
        point()
          : x_(0), y_(0)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(double x, double y) 
        {
            x_ = x;
            y_ = y;
        }

        /// retrieve the X coordinate of this point
        double get_X() const
        {
            return x_;
        }

        /// retrieve the Y coordinate of this point
        double get_Y() const
        {
            return y_;
        }

        /// modify the X coordinate of this point
        void set_X(double x) 
        {
            x_ = x;
        }

        /// modify the Y coordinate of this point
        void set_Y(double y) 
        {
            y_ = y;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_action2<
            point, point_init, double, double, &point::init
        > init_action;

        typedef hpx::actions::direct_result_action0<
            point const, double, point_get_X, &point::get_X
        > get_X_action;

        typedef hpx::actions::direct_result_action0<
            point const, double, point_get_Y, &point::get_Y
        > get_Y_action;

        typedef hpx::actions::direct_action1<
            point, point_set_X, double, &point::set_X
        > set_X_action;

        typedef hpx::actions::direct_action1<
            point, point_set_Y, double, &point::set_Y
        > set_Y_action;

    private:
        double x_, y_;
    };

}}}

#endif
