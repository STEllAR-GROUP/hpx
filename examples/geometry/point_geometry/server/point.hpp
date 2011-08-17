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

typedef boost::geometry::model::d2::point_xy<double> point_type;
typedef boost::geometry::model::polygon<point_type> polygon_type;

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
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
            point_search = 5,
            point_get_poly = 6,
            point_move = 7
        };

        // constructor: initialize accumulator value
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(double xmin,double xmax,double ymin,double ymax,double velx, double vely, std::size_t numpoints) 
        {
            xmin_ = xmin;
            xmax_ = xmax;
            ymin_ = ymin;
            ymax_ = ymax;
            numpoints_ = numpoints;
  
            double dx = (xmax - xmin)/(numpoints-1);
            double dy = (ymax - ymin)/(numpoints-1);

            //typedef bg::model::d2::point_xy<double> point_type;
            //typedef bg::model::polygon<point_type> polygon_type;

            //polygon_type poly;
            // create the rectangle of the mesh object
            for (std::size_t i=0;i<numpoints;i++) {
              double x = xmin + dx*i;
              point_type p(x,ymin);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=0;i<numpoints;i++) {
              double y = ymin + dy*i;
              point_type p(xmax,y);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=0;i<numpoints;i++) {
              double x = xmax - dx*i;
              point_type p(x,ymax);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=0;i<numpoints;i++) {
              double y = ymax - dy*i;
              point_type p(xmin,y);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }

            //point_type pt5(0.5, 0.5);
            //bool inside = bg::within(pt5,poly_);
            //std::cout << " Point is " << (inside ? "inside" : "outside") << std::endl;

            //pt_.x(x);
            //pt_.y(y);
        }

        /// search for contact
        int search(std::vector<hpx::naming::id_type> const& search_objects);

        /// callback for search-wait
        bool search_callback(std::size_t idx, polygon_type const& poly,bool &redo);

        // move the bodies
        void move(double dt);

        // retrieve the polygon object
        polygon_type get_poly() const
        {
            return poly_;
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
        typedef hpx::actions::direct_action7<
            point, point_init, double, double,double,double,double,double,std::size_t, &point::init
        > init_action;

        typedef hpx::actions::direct_result_action0<
            point const, double, point_get_X, &point::get_X
        > get_X_action;

        typedef hpx::actions::direct_result_action0<
            point const, polygon_type, point_get_poly, &point::get_poly
        > get_poly_action;

        typedef hpx::actions::direct_action1<
            point, point_move, double, &point::move
        > move_action;

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
            point, int, point_search, std::vector<hpx::naming::id_type> const&, 
            &point::search
        > search_action;

    private:
        plain_point_type pt_;
        double xmin_,xmax_,ymin_,ymax_;
        std::size_t numpoints_;
        polygon_type poly_;
        std::vector<double> velx_;
        std::vector<double> vely_;
        // for contact enforcement
        std::vector<std::size_t> slave_;
        std::vector<std::size_t> master_;
        std::vector<std::size_t> object_id_;
        //boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > p_;
    };

}}}

#endif
