//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_POINT)
#define HPX_COMPONENTS_STUBS_POINT

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

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
        static lcos::promise<void>
        init_async(naming::id_type gid, double xmin, double xmax,double ymin,double ymax,double velx,double vely,std::size_t numpoints,std::size_t objectid)
        {
            typedef server::point::init_action action_type;
            return lcos::eager_future<action_type>(gid,xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid);
        }

        static void init(naming::id_type const& gid,double xmin, double xmax,double ymin,double ymax,double velx, double vely,std::size_t numpoints,std::size_t objectid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            init_async(gid,xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid).get();
        }

        static lcos::promise<int>
        search_async(naming::id_type gid, std::vector<hpx::naming::id_type> const& search_objects)
        {
            typedef server::point::search_action action_type;
            return lcos::eager_future<action_type>(gid, search_objects);
        }

        static lcos::promise<void>
        recompute_async(naming::id_type gid, std::vector<hpx::naming::id_type> const& search_objects)
        {
            typedef server::point::recompute_action action_type;
            return lcos::eager_future<action_type>(gid, search_objects);
        }

        static int search(naming::id_type const& gid, std::vector<hpx::naming::id_type> const& search_objects)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return search_async(gid, search_objects).get();
        }

        static void recompute(naming::id_type const& gid, std::vector<hpx::naming::id_type> const& search_objects)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            recompute_async(gid, search_objects).get();
        }

        static lcos::promise<polygon_type>
        get_poly_async(naming::id_type gid)
        {
            typedef server::point::get_poly_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static polygon_type get_poly(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return get_poly_async(gid).get();
        }

        static lcos::promise<void>
        move_async(naming::id_type gid,double dt,double time)
        {
            typedef server::point::move_action action_type;
            return lcos::eager_future<action_type>(gid,dt,time);
        }

        static lcos::promise<void>
        adjust_async(naming::id_type gid,double dt)
        {
            typedef server::point::adjust_action action_type;
            return lcos::eager_future<action_type>(gid,dt);
        }

        static lcos::promise<void>
        enforce_async(naming::id_type gid,std::vector<hpx::naming::id_type> const& master_gids,double dt,
                      std::size_t n,std::size_t N)
        {
            typedef server::point::enforce_action action_type;
            return lcos::eager_future<action_type>(gid,master_gids,dt,n,N);
        }

        static void move(naming::id_type const& gid,double dt,double time)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            move_async(gid,dt,time).get();
        }

        static void adjust(naming::id_type const& gid,double dt)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            adjust_async(gid,dt).get();
        }

        static void enforce(naming::id_type const& gid,std::vector<hpx::naming::id_type> const& master_gids,double dt,
                            std::size_t n,std::size_t N)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            enforce_async(gid,master_gids,dt,n,N).get();
        }

        /// Query the current coordinate values of the server#point
        /// instance with the given \a gid.
        static lcos::promise<double>
        get_X_async(naming::id_type const& gid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::get_X_action action_type;
            return lcos::eager_future<action_type>(gid);
        }
        static lcos::promise<double>
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
            // is executed and the result is returned to the promise
            return get_X_async(gid).get();
        }
        static double get_Y(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return get_Y_async(gid).get();
        }

        /// Modify the current coordinate values of the server#point
        /// instance with the given \a gid.
        static lcos::promise<void>
        set_X_async(naming::id_type const& gid, double x)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized eager_future, the caller needs
            // to call get() on the return value to obtain the result
            typedef server::point::set_X_action action_type;
            return lcos::eager_future<action_type>(gid, x);
        }
        static lcos::promise<void>
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
            // is executed and the result is returned to the promise
            set_X_async(gid, x).get();
        }
        static void set_Y(naming::id_type const& gid, double y)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            set_Y_async(gid, y).get();
        }
    };

}}}

#endif
