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

            return 0;
        }

        bool point::search_callback(std::size_t i, polygon_type const& poly,bool &redo) 
        {

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
            std::cout << i << " Contact region area  " << boost::geometry::area(p) << std::endl;
            std::cout << " contact region size " << p.outer().size() << std::endl;

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
                  for (std::size_t k=0;k<poly_.outer().size();k++) {
                    if ( boost::geometry::distance( (p.outer())[j],(poly_.outer())[k] ) < 1.e-10 ) {
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
                  for (std::size_t k=0;k<poly_.outer().size();k++) {
                    final = k+1; 
                    if ( k+1 >= poly_.outer().size() ) final = 0;
                    line[0] = (poly_.outer())[k];
                    line[1] = (poly_.outer())[final];
                    double testdist = boost::geometry::distance((p.outer())[j],line);    
                    if ( mindist > testdist ) {
                      testdist = mindist;
                      min_k = k;
                    }
                  }
                  BOOST_ASSERT(min_k >= 0 );
                  master_.push_back(min_k); 
                }
              }
            }
          }

          // for each node in the polygon, see if there is contact with this polygon
          //for (std::size_t j=0;j<poly_.outer().size();j++) {
          //  if ( boost::geometry::within(poly.outer()[j],poly) ) {
            // Contact!
            // Find the master segment -- the two nodes of the polygon 
            //                             nearest the contact node

            // Record pertinent information for contact enforcement
            //    -Master segment
            //    -l,A,B,C,delta, R_1, R_2,xsm,zsm
               
          //  }
          //}
          //std::cout << " TEST poly.outer().size() " << poly.outer().size() << " i " << i << " poly_ " << poly_.outer().size() << std::endl;

          // return type says continue or not
          // usually return true
          return true;
        }

        void point::move(double dt)
        {
          for (std::size_t i=0;i<poly_.outer().size();i++) {
            point_type &p = (poly_.outer())[i];
            (poly_.outer())[i].x(p.x() + velx_[i]*dt);
            (poly_.outer())[i].y(p.y() + vely_[i]*dt);
          }
        }
}}}

