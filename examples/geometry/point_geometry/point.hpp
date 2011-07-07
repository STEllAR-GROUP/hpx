//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_point)
#define HPX_COMPONENTS_CLIENT_point

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/register/point.hpp>

#include "stubs/point.hpp"

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
          : base_type(base_type::create_sync(where))    // create component
        {
            init(x, y);   // initialize coordinates
        }

        /// Create a client side representation for the existing
        /// \a server#point instance with the given global id \a gid.
        point(naming::id_type gid) 
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#point instance with the given \a gid
        lcos::future_value<void> init_async(double x, double y) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_, x, y);
        }

        void init(double x, double y) 
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_, x, y);
        }

        /// Query the current coordinate values of the server#point 
        /// instance with the given \a gid. 
        lcos::future_value<double> get_X_async() const
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_X_async(gid_);
        }
        lcos::future_value<double> get_Y_async()  const
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
        lcos::future_value<void> set_X_async(double x) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::set_X_async(gid_, x);
        }
        lcos::future_value<void> set_Y_async(double y) 
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
