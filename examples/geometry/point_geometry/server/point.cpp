//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include "../serialize_geometry.hpp"
#include "../stubs/point.hpp"
#include "./point.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server 
{
        /// search for contact
        int point::search(std::vector<hpx::naming::id_type> const& search_objects) 
        {
            typedef std::vector<lcos::future_value<polygon_type> > lazy_results_type;

            lazy_results_type lazy_results;
            BOOST_FOREACH(naming::id_type gid, search_objects)
            {
              lazy_results.push_back( stubs::point::get_poly_async( gid ) );
            }

            // will return the number of invoked futures
            bool redo = false;
            components::wait(lazy_results, boost::bind(&point::search_callback, this, _1, _2,boost::ref(redo)));

            if ( slave_.size() > 0 ) return 1;
            else return 0;
        }

        bool point::search_callback(std::size_t i, polygon_type const& poly,bool &redo) 
        {

          //std::cout << " TEST in callback " << i << " object id " << objectid_ << std::endl;

          if ( i != objectid_ ) {
            // search for contact

            // TEST 
            //std::cout << " TEST search callback " << std::endl;
            //for (std::size_t j=0;j<poly_.outer().size();j++) {
            //  std::cout << (poly_.outer())[j].x() << " " << (poly_.outer())[j].y() << std::endl;
            //}
            //std::cout << " END TEST search callback " << std::endl << std::endl;

            typedef boost::geometry::model::linestring<point_type> linestring_type;

            // Check for contact
            std::deque<polygon_type> output;
            boost::geometry::intersection(poly_,poly,output);
            BOOST_FOREACH(polygon_type const& p, output) {
             // std::cout << i << " Contact region area  " << boost::geometry::area(p) << std::endl;
             // std::cout << " contact region size " << p.outer().size() << std::endl;

              linestring_type line;
              line.resize(2);

              for (std::size_t j=0;j<p.outer().size();j++) {
                double d1 = boost::geometry::distance((p.outer())[j],poly_);
                double d2 = boost::geometry::distance((p.outer())[j],poly);
                // Check if there is actual contact
                if ( d1 > 1.e-10 || d2 > 1.e-10 ) {
                  // there is actual contact -- find the vertices of poly_
                  // which have contact and their corresponding master segments 
                  if ( d1 < 1.e-10 ) {
                    // this is a vertex belonging to poly_
                    // record the point as a slave
                    // find the index of poly_ that corresponds to this point
                    point_type const& pp = (p.outer())[j];
                    polygon_type::ring_type const& outer = poly_.outer();
                    for (std::size_t k=0;k<poly_.outer().size();k++) {
                      if ( boost::geometry::distance(pp,outer[k]) < 1.e-10 ) {
                        slave_.push_back(k); 
                        break;
                      }
                    }

                    // The following section will be replaced by Barend 
                    // with the nearest neighbor routine
                    // but for now, this works

                    // the master segment should belong to poly
                    object_id_.push_back(i);        
                    double mindist = 999.;
                    int min_k = -1;
                    int final;
                    for (std::size_t k=0;k<poly.outer().size();k++) {
                      final = k+1; 
                      if ( k+1 >= poly.outer().size() ) final = 0;
                      line[0] = (poly.outer())[k];
                      line[1] = (poly.outer())[final];
                      double testdist = boost::geometry::distance(pp,line);    
                      if ( testdist < mindist ) {
                        mindist = testdist;
                        min_k = k;
                    //    std::cout << "        Search slave x : " << pp.x()  << " y " << pp.y() << std::endl;
                    //    std::cout << "        Search master1 x : " << (poly.outer())[k].x()  << " y " << (poly.outer())[k].y() << std::endl;
                    //    std::cout << "        Search master2 x : " << (poly.outer())[final].x()  << " y " << (poly.outer())[final].y() << std::endl;
                      }
                    }
                    BOOST_ASSERT(min_k >= 0 );
                    master_.push_back(min_k); 
                  }
                }
              }
            }

            // TEST
            for (std::size_t j=0;j<slave_.size();j++) {
              std::cout << " TEST slave : " << slave_[j] << " master " << master_[j] << " object " << object_id_[j] << std::endl; 
              std::cout << "        Follow up slave x : " << (poly_.outer())[slave_[j]].x()  << " y " << (poly_.outer())[slave_[j]].y() << std::endl;
              std::cout << "        Follow up master x : " << (poly.outer())[master_[j]].x()  << " y " << (poly.outer())[master_[j]].y() << std::endl;
            }

          }

          // return type says continue or not
          // usually return true
          return true;
        }

        void point::move(double dt)
        {
          // clear out slave nodes and master segments
          slave_.resize(0);
          master_.resize(0);
          object_id_.resize(0);
          for (std::size_t i=0;i<poly_.outer().size();i++) {
            point_type &p = (poly_.outer())[i];
            (poly_.outer())[i].x(p.x() + velx_[i]*dt);
            (poly_.outer())[i].y(p.y() + vely_[i]*dt);
          }
        }

        void point::enforce(std::vector<hpx::naming::id_type> const& master_gids)
        {
          std::cout << " HELLO WORLD TEST : " << objectid_ << " " << slave_.size() << std::endl;
 
          std::vector<hpx::lcos::future_value<vertex_data> > iterate_phase;
          vertex_data slave;
          std::size_t master_vertex,master_object;
          for (std::size_t i=0;i<slave_.size();i++) {
            slave.x = (poly_.outer())[slave_[i]].x();    
            slave.y = (poly_.outer())[slave_[i]].y();   
            slave.velx = velx_[i];    
            slave.vely = vely_[i];    
            master_vertex = master_[i];
            master_object = object_id_[i];
            //iterate_phase.push_back(accu[master_object].iterate_async(slave,master_vertex)); 
          }
          //hpx::components::wait(iterate_phase);
        }

        vertex_data point::iterate(vertex_data slave,std::size_t master_vertex)
        {
          std::cout << " HELLO WORLD TEST iterate master " << objectid_ << std::endl;
          // iterate on the master vertices, return the updated slave point
          // This section comes from Eqns 4-29 in the Johnson and Stryk paper
          return slave;
        }

}}}

