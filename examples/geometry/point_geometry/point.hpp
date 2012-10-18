//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_POINT)
#define HPX_COMPONENTS_CLIENT_POINT

#include <hpx/hpx.hpp>
#include <hpx/include/client.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/register/point.hpp>

#include "stubs/point.hpp"

typedef boost::geometry::model::d2::point_xy<double> point_type;
typedef boost::geometry::model::polygon<point_type> polygon_type;

namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#point class is the client side representation of all
    /// \a server#point components
    class point
        : public components::client_base<point, stubs::point>
    {
        typedef components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component)
        point()
        {}

        /// Create client side representation from a newly create component
        /// instance.
        point(naming::id_type where, double x, double y)
          : base_type(base_type::create(where))    // create component
        {
            //init(x, y);   // initialize coordinates
        }

        /// Create a client side representation for the existing
        /// \a server#point instance with the given global id \a gid.
        point(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#point instance with the given \a gid
        lcos::future<void> init_async(double xmin, double xmax,
                                            double ymin, double ymax,
                                            double velx, double vely,
                                            std::size_t numpoints,
                                            std::size_t objectid)
        {
            return this->base_type::init_async(get_gid(), xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid);
        }

        void init(double xmin, double xmax,
                  double ymin, double ymax,
                  double velx, double vely,
                  std::size_t numpoints,std::size_t objectid)
        {
            this->base_type::init_async(get_gid(),xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid);
        }

        /// Initialize the server#point instance with the given \a gid
        lcos::future<int> search_async(std::vector<hpx::naming::id_type> const& search_objects)
        {
            return this->base_type::search_async(get_gid(), search_objects);
        }

        lcos::future<void> recompute_async(std::vector<hpx::naming::id_type> const& search_objects)
        {
            return this->base_type::recompute_async(get_gid(), search_objects);
        }

        int search(std::vector<hpx::naming::id_type> const& search_objects)
        {
            return this->base_type::search(get_gid(), search_objects);
        }

        void recompute(std::vector<hpx::naming::id_type> const& search_objects)
        {
            this->base_type::recompute(get_gid(), search_objects);
        }

        lcos::future<polygon_type> get_poly_async() const
        {
            return this->base_type::get_poly_async(get_gid());
        }

        polygon_type get_poly() const
        {
            return this->base_type::get_poly(get_gid());
        }

        lcos::future<void> move_async(double dt,double time)
        {
            return this->base_type::move_async(get_gid(),dt,time);
        }

        lcos::future<void> adjust_async(double dt)
        {
            return this->base_type::adjust_async(get_gid(),dt);
        }

        lcos::future<void> enforce_async(std::vector<hpx::naming::id_type> const& master_gids,double dt,
                                               std::size_t n,std::size_t N)
        {
            return this->base_type::enforce_async(get_gid(),master_gids,dt,n,N);
        }

        void move(double dt,double time)
        {
            this->base_type::move(get_gid(),dt,time);
        }

        void adjust(double dt)
        {
            this->base_type::adjust(get_gid(),dt);
        }

        void enforce(std::vector<hpx::naming::id_type> const& master_gids,double dt,
                     std::size_t n,std::size_t N)
        {
            this->base_type::enforce(get_gid(),master_gids,dt,n,N);
        }

        /// Query the current coordinate values of the server#point
        /// instance with the given \a gid.
        lcos::future<double> get_X_async() const
        {
            return this->base_type::get_X_async(get_gid());
        }
        lcos::future<double> get_Y_async()  const
        {
            return this->base_type::get_Y_async(get_gid());
        }

        double get_X() const
        {
            return this->base_type::get_X(get_gid());
        }
        double get_Y() const
        {
            return this->base_type::get_Y(get_gid());
        }

        /// Modify the current coordinate values of the server#point
        /// instance with the given \a gid.
        lcos::future<void> set_X_async(double x)
        {
            return this->base_type::set_X_async(get_gid(), x);
        }
        lcos::future<void> set_Y_async(double y)
        {
            return this->base_type::set_Y_async(get_gid(), y);
        }

        void set_X(double x)
        {
            this->base_type::set_X(get_gid(), x);
        }
        void set_Y(double y)
        {
            this->base_type::set_Y(get_gid(), y);
        }
    };
}}

// Adapt this point type to Boost.Geometry. This allows to use this type
// wherever Boost.Geometry is expecting a point type.
BOOST_GEOMETRY_REGISTER_POINT_2D_GET_SET(
    hpx::geometry::point, double, boost::geometry::cs::cartesian,
    get_X, get_Y, set_X, set_Y)

#endif
