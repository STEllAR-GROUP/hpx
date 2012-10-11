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
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_, xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid);
        }

        void init(double xmin, double xmax,
                  double ymin, double ymax,
                  double velx, double vely,
                  std::size_t numpoints,std::size_t objectid)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,xmin,xmax,ymin,ymax,velx,vely,numpoints,objectid);
        }

        /// Initialize the server#point instance with the given \a gid
        lcos::future<int> search_async(std::vector<hpx::naming::id_type> const& search_objects)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::search_async(gid_, search_objects);
        }

        lcos::future<void> recompute_async(std::vector<hpx::naming::id_type> const& search_objects)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::recompute_async(gid_, search_objects);
        }

        int search(std::vector<hpx::naming::id_type> const& search_objects)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::search(gid_, search_objects);
        }

        void recompute(std::vector<hpx::naming::id_type> const& search_objects)
        {
            BOOST_ASSERT(gid_);
            this->base_type::recompute(gid_, search_objects);
        }

        lcos::future<polygon_type> get_poly_async() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_poly_async(gid_);
        }

        polygon_type get_poly() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_poly(gid_);
        }

        lcos::future<void> move_async(double dt,double time)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::move_async(gid_,dt,time);
        }

        lcos::future<void> adjust_async(double dt)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::adjust_async(gid_,dt);
        }

        lcos::future<void> enforce_async(std::vector<hpx::naming::id_type> const& master_gids,double dt,
                                               std::size_t n,std::size_t N)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::enforce_async(gid_,master_gids,dt,n,N);
        }

        void move(double dt,double time)
        {
            BOOST_ASSERT(gid_);
            this->base_type::move(gid_,dt,time);
        }

        void adjust(double dt)
        {
            BOOST_ASSERT(gid_);
            this->base_type::adjust(gid_,dt);
        }

        void enforce(std::vector<hpx::naming::id_type> const& master_gids,double dt,
                     std::size_t n,std::size_t N)
        {
            BOOST_ASSERT(gid_);
            this->base_type::enforce(gid_,master_gids,dt,n,N);
        }

        /// Query the current coordinate values of the server#point
        /// instance with the given \a gid.
        lcos::future<double> get_X_async() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_X_async(gid_);
        }
        lcos::future<double> get_Y_async()  const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_Y_async(gid_);
        }

        double get_X() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_X(gid_);
        }
        double get_Y() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_Y(gid_);
        }

        /// Modify the current coordinate values of the server#point
        /// instance with the given \a gid.
        lcos::future<void> set_X_async(double x)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::set_X_async(gid_, x);
        }
        lcos::future<void> set_Y_async(double y)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::set_Y_async(gid_, y);
        }

        void set_X(double x)
        {
            BOOST_ASSERT(gid_);
            this->base_type::set_X(gid_, x);
        }
        void set_Y(double y)
        {
            BOOST_ASSERT(gid_);
            this->base_type::set_Y(gid_, y);
        }
    };
}}

// Adapt this point type to Boost.Geometry. This allows to use this type
// wherever Boost.Geometry is expecting a point type.
BOOST_GEOMETRY_REGISTER_POINT_2D_GET_SET(
    hpx::geometry::point, double, boost::geometry::cs::cartesian,
    get_X, get_Y, set_X, set_Y)

#endif
