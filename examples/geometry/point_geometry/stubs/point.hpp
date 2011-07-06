//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_point)
#define HPX_COMPONENTS_STUBS_point

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "server/point.hpp"

namespace hpx { namespace geometry { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#point class is the client side representation of all
    /// \a server#point components
    struct point : components::stub_base<server::point>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#point instance with the given \a gid
        static lcos::future_value<void> 
        init_async(naming::id_type gid, double x, double y) 
        {
            typedef server::point::init_action action_type;
            return lcos::eager_future<action_type>(gid, x, y);
        }

        static void init(naming::id_type const& gid, double x, double y) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            init_async(gid, x, y).get();
        }

        /// Query the current coordinate values of the server#point 
        /// instance with the given \a gid. 
        static lcos::future_value<double> 
        get_X_async(naming::id_type const& gid) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::get_X_action action_type;
            return lcos::eager_future<action_type>(gid);
        }
        static lcos::future_value<double> 
        get_Y_async(naming::id_type const& gid) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::get_Y_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static double get_X(naming::id_type const& gid) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            return get_X_async(gid).get();
        }
        static double get_Y(naming::id_type const& gid) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            return get_Y_async(gid).get();
        }

        /// Modify the current coordinate values of the server#point 
        /// instance with the given \a gid. 
        static lcos::future_value<void> 
        set_X_async(naming::id_type const& gid, double x) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::set_X_action action_type;
            return lcos::eager_future<action_type>(gid, x);
        }
        static lcos::future_value<void> 
        set_Y_async(naming::id_type const& gid, double y) 
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::set_Y_action action_type;
            return lcos::eager_future<action_type>(gid, y);
        }


        static void set_X(naming::id_type const& gid, double x) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            set_X_async(gid, x).get();
        }
        static void set_Y(naming::id_type const& gid, double y) 
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the future_value
            set_Y_async(gid, y).get();
        }
    };

}}}

#endif
