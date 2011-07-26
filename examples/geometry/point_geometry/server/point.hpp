//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_POINT)
#define HPX_COMPONENTS_SERVER_POINT

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>

#include "../serialize_geometry.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
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
            point_set_Y = 4,
            point_search = 5
        };

        // constructor: initialize accumulator value
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(double xmin,double xmax,double ymin,double ymax,std::size_t numpoints) 
        {
            namespace bg = boost::geometry;

            xmin_ = xmin;
            xmax_ = xmax;
            ymin_ = ymin;
            ymax_ = ymax;
            numpoints_ = numpoints;
  
            double dx = (xmax - xmin)/(numpoints-1);
            double dy = (ymax - ymin)/(numpoints-1);

            bg::model::d2::point_xy<int> p1(1, 1), p2(2, 2);
            std::cout << "Distance p1-p2 is: " << bg::distance(p1, p2) << std::endl;

            double points[][2] = {{2.0, 1.3}, {4.1, 3.0}, {5.3, 2.6}, {2.9, 0.7}, {2.0, 1.3}};
            bg::model::polygon<bg::model::d2::point_xy<double> > poly;
            bg::append(poly, points);
#if 0
            double pt[2] = {xmin,ymin};            
            boost::geometry::append(p_,pt);
            // create the rectangle of the mesh object
            for (std::size_t i=0;i<numpoints;i++) {
              double x = xmin + dx*i;
              double pt[2] = {x,ymin};            
              p_.outer().push_back(pt);            
            }
            for (std::size_t i=0;i<numpoints;i++) {
              double y = ymin + dy*i;
              double pt[2] = {xmax,y};            
              p_.outer().push_back(pt);            
            }
            for (std::size_t i=numpoints-1;i>=0;i--) {
              double x = xmin + dx*i;
              double pt[2] = {x,ymax};            
              p_.outer().push_back(pt);            
            }
            for (std::size_t i=numpoints-1;i>=0;i--) {
              double y = ymin + dy*i;
              double pt[2] = {xmin,y};            
              p_.outer().push_back(pt);            
            }
#endif

            //pt_.x(x);
            //pt_.y(y);
        }

        /// search for contact
        bool search(plain_polygon_type const& p) const
        {
            return boost::geometry::within(pt_, p);
        }

        /// retrieve the X coordinate of this point
        double get_X() const
        {
            return pt_.x();
        }

        /// retrieve the Y coordinate of this point
        double get_Y() const
        {
            return pt_.y();
        }

        /// modify the X coordinate of this point
        void set_X(double x) 
        {
            pt_.x(x);
        }

        /// modify the Y coordinate of this point
        void set_Y(double y) 
        {
            pt_.y(y);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_action5<
            point, point_init, double, double,double,double,std::size_t, &point::init
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

        typedef hpx::actions::direct_result_action1<
            point const, bool, point_search, plain_polygon_type const&, 
            &point::search
        > search_action;

    private:
        plain_point_type pt_;
        double xmin_,xmax_,ymin_,ymax_;
        std::size_t numpoints_;
        boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > p_;
    };

}}}

#endif
