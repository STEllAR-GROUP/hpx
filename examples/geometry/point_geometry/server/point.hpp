//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_POINT)
#define HPX_COMPONENTS_SERVER_POINT

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include "../serialize_geometry.hpp"

typedef boost::geometry::model::d2::point_xy<double> point_type;
typedef boost::geometry::model::polygon<point_type> polygon_type;

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server
{

    struct vertex_data
    {
      double x,y,velx,vely;

      private:
        // serialization support
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & x & y & velx & vely;
        }

    };

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public components::managed_component_base<point>
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
            point_move = 7,
            point_enforce = 8,
            point_adjust = 9,
            point_recompute = 10
        };

        // constructor: initialize accumulator value
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(double xmin,double xmax,double ymin,double ymax,double velx, double vely, std::size_t numpoints,std::size_t objectid)
        {
            xmin_ = xmin;
            xmax_ = xmax;
            ymin_ = ymin;
            ymax_ = ymax;
            numpoints_ = numpoints;
            objectid_ = objectid;

            double dx = (xmax - xmin)/(numpoints-1);
            double dy = (ymax - ymin)/(numpoints-1);

            //typedef bg::model::d2::point_xy<double> point_type;
            //typedef bg::model::polygon<point_type> polygon_type;

            // Add points *CLOCKWISE* -- otherwise intersection routine will not work
            //polygon_type poly;
            // create the rectangle of the mesh object

            // Pre-allocate storage for the vectors, so that we avoid re-allocations
            // during the initialization loops.
            velx_.reserve(numpoints*4);
            vely_.reserve(numpoints*4);
            poly_.outer().reserve(numpoints*4);

            for (std::size_t i=0;i<numpoints;i++) {
              double y = ymin + dy*i;
              point_type p(xmin,y);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=1;i<numpoints;i++) {
              double x = xmin + dx*i;
              point_type p(x,ymax);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=1;i<numpoints;i++) {
              double y = ymax - dy*i;
              point_type p(xmax,y);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }
            for (std::size_t i=1;i<numpoints-1;i++) {
              double x = xmax - dx*i;
              point_type p(x,ymin);
              poly_.outer().push_back(p);
              velx_.push_back(velx);
              vely_.push_back(vely);
            }

            // Close the polygon -- this will add one more point
            boost::geometry::correct(poly_);

            // give a velocity for the last (duplicated) point
            velx_.push_back(velx);
            vely_.push_back(vely);

            BOOST_ASSERT( velx_.size() == vely_.size() && velx_.size() == poly_.outer().size() );

            // Correct any problems
            boost::geometry::correct(poly_);

            BOOST_ASSERT(boost::geometry::area(poly_) > 0);

            //for (std::size_t i=0;i<poly_.outer().size();i++) {
            //  std::cout << (poly_.outer())[i].x() << " " << (poly_.outer())[i].y() << std::endl;
            //}

        }

        /// search for contact
        int search(std::vector<hpx::naming::id_type> const& search_objects);

        /// Recompute Rsum
        void recompute(std::vector<hpx::naming::id_type> const& search_objects);

        /// callback for search-wait
        bool search_callback(std::size_t idx, polygon_type const& poly,bool &redo);

        // move the bodies
        void move(double dt,double time);

        // move the bodies
        void adjust(double dt);

        // enforce the contact
        void enforce(std::vector<hpx::naming::id_type> const& master_gids,double dt,std::size_t n,std::size_t N);

        // what to do on the slave node when the master segment component finishes
        // iterating
        bool enforce_callback(std::size_t i, polygon_type const& poly,double dt,std::size_t n,std::size_t N);

        bool recompute_callback(std::size_t i, polygon_type const& poly);

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
        typedef hpx::actions::action8<
            point, point_init, double, double,double,double,double,double,std::size_t,std::size_t, &point::init
        > init_action;

        typedef hpx::actions::result_action0<
            point const, double, point_get_X, &point::get_X
        > get_X_action;

        typedef hpx::actions::result_action0<
            point const, polygon_type, point_get_poly, &point::get_poly
        > get_poly_action;

        typedef hpx::actions::action2<
            point, point_move, double, double, &point::move
        > move_action;

        typedef hpx::actions::action1<
            point, point_adjust, double, &point::adjust
        > adjust_action;

        typedef hpx::actions::action4<
            point, point_enforce,std::vector<hpx::naming::id_type> const&,double,std::size_t,std::size_t,
            &point::enforce
        > enforce_action;

        typedef hpx::actions::result_action0<
            point const, double, point_get_Y, &point::get_Y
        > get_Y_action;

        typedef hpx::actions::action1<
            point, point_set_X, double, &point::set_X
        > set_X_action;

        typedef hpx::actions::action1<
            point, point_set_Y, double, &point::set_Y
        > set_Y_action;

        typedef hpx::actions::result_action1<
            point, int, point_search, std::vector<hpx::naming::id_type> const&,
            &point::search
        > search_action;

        typedef hpx::actions::action1<
            point, point_recompute, std::vector<hpx::naming::id_type> const&,
            &point::recompute
        > recompute_action;

    private:
        //hpx::lcos::local::mutex mtx_;    // lock for this data block
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
        std::vector<std::size_t> inv_slave_;
        std::vector<std::size_t> inv_master_;
        std::vector<std::size_t> inv_object_id_;
        std::vector<double> R_;
        std::vector<double> change_vx_;
        std::vector<double> change_vy_;
        std::size_t objectid_;
        //boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > p_;
    };

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::init_action,
    point_geometry_init_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::move_action,
    point_geometry_move_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::adjust_action,
    point_geometry_adjust_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::enforce_action,
    point_geometry_enforce_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::search_action,
    point_geometry_search_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::recompute_action,
    point_geometry_recompute_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::get_poly_action,
    point_geometry_get_poly_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::get_X_action,
    point_geometry_get_X_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::get_Y_action,
    point_geometry_get_Y_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::set_X_action,
    point_geometry_set_X_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::geometry::server::point::set_Y_action,
    point_geometry_set_Y_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<polygon_type>::set_value_action,
    set_value_action_polygon_type);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<hpx::geometry::server::vertex_data>::set_value_action,
    set_value_action_vertex_data_type);

#endif
