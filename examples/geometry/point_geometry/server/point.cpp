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

#include <iostream>
#include <fstream>

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
            typedef boost::geometry::model::linestring<point_type> linestring_type;

            // Check for contact
            std::deque<polygon_type> output;
            boost::geometry::intersection(poly_,poly,output);
            BOOST_FOREACH(polygon_type const& p, output) {
              linestring_type line;
              line.resize(2);

              char basename[80];
              sprintf(basename,"overlap%d_%d.dat",(int) objectid_,(int) i);
              std::ofstream file;
              file.open(basename);
              // note that the first and the last vertices are the same
              for (std::size_t j=0;j<p.outer().size()-1;j++) {
                file << (p.outer())[j].x() << " " << (p.outer())[j].y() << std::endl;

                point_type const& pp = (p.outer())[j];

                // find the vertex which belongs to poly_
                // record the point as a slave
                // find the index of poly_ that corresponds to this point
                polygon_type::ring_type const& outer = poly_.outer();
                bool found = false;
                for (std::size_t k=0;k<poly_.outer().size();k++) {
                  if ( boost::geometry::distance(pp,outer[k]) < 1.e-10 ) {
                    slave_.push_back(k); 
                    found = true;
                    break;
                  }
                }
                if ( found ) {
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
                    }
                  }

                  BOOST_ASSERT(min_k >= 0 );
                  master_.push_back(min_k); 
                }
              }
              file.close();
            }

            // TEST
            //for (std::size_t j=0;j<slave_.size();j++) {
            //  std::cout << " TEST slave : " << slave_[j] << " master " << master_[j] << " object " << object_id_[j] << std::endl; 
            //  std::cout << "        Follow up slave x : " << (poly_.outer())[slave_[j]].x()  << " y " << (poly_.outer())[slave_[j]].y() << std::endl;
            //  std::cout << "        Follow up master1 x : " << (poly.outer())[master_[j]].x()  << " y " << (poly.outer())[master_[j]].y() << std::endl;
            //  std::size_t final = master_[j] + 1; 
            //  if ( final >= poly.outer().size() ) final = 0;
            //  std::cout << "        Follow up master2 x : " << (poly.outer())[final].x()  << " y " << (poly.outer())[final].y() << std::endl;
            //}

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

          char basename[80];
          sprintf(basename,"geo%d.dat",(int) objectid_);
          std::ofstream file;
          file.open(basename);
          for (std::size_t i=0;i<poly_.outer().size();i++) {
            file << (poly_.outer())[i].x() << " " << (poly_.outer())[i].y() << std::endl;
          }
          file.close();
        }

        void point::enforce(std::vector<hpx::naming::id_type> const& master_gids)
        {
          typedef std::vector<lcos::future_value<vertex_data> > lazy_results_type;

          lazy_results_type lazy_results;
 
          std::vector<hpx::lcos::future_value<vertex_data> > iterate_phase;
          vertex_data slave;
          std::size_t master_vertex;
          for (std::size_t i=0;i<slave_.size();i++) {
            slave.x = (poly_.outer())[slave_[i]].x();    
            slave.y = (poly_.outer())[slave_[i]].y();   
            slave.velx = velx_[i];    
            slave.vely = vely_[i];    
            master_vertex = master_[i];
            naming::id_type gid = master_gids[ object_id_[i] ];
            lazy_results.push_back( stubs::point::iterate_async( gid,slave,master_vertex ) );
          }

          // will return the number of invoked futures
          components::wait(lazy_results, boost::bind(&point::enforce_callback, this, _1, _2));
        }

        bool point::enforce_callback(std::size_t i, vertex_data const& slave) 
        {
          // This is where you update the slave node
          (poly_.outer())[slave_[i]].x();
          (poly_.outer())[i].x(slave.x);
          (poly_.outer())[i].y(slave.y);
          velx_[i] = slave.velx;
          vely_[i] = slave.vely;
          // return type says continue or not
          // usually return true
          return true;
        }

        vertex_data point::iterate(vertex_data slave,std::size_t master_vertex)
        {
          hpx::lcos::mutex::scoped_lock lock(mtx_);
         
          std::size_t final = master_vertex+1; 
          if ( final >= poly_.outer().size() ) final = 0;

          point_type pp;

          pp.x(slave.x);
          pp.y(slave.y);

          double l = boost::geometry::distance((poly_.outer())[master_vertex],(poly_.outer())[final]);

          typedef boost::geometry::model::linestring<point_type> linestring_type;
          linestring_type line;
          line.resize(2);
          line[0] = (poly_.outer())[master_vertex];
          line[1] = (poly_.outer())[final];
          double tdelta = boost::geometry::distance(pp,line);

          double x1 = (poly_.outer())[master_vertex].x();
          double x2 = (poly_.outer())[final].x();
          double z1 = (poly_.outer())[master_vertex].y();
          double z2 = (poly_.outer())[final].y();

          double A = (z2-z1)/l;
          double B = (x1-x2)/l;
          double C = (x2*z1-x1*z2)/l;

          double xs = slave.x;
          double zs = slave.y;
          double delta = -(A*xs + B*zs + C);

          double xsm = xs + A*delta;
          double zsm = zs + B*delta;

          double R_1 = sqrt( (xsm-x2)*(xsm-x2) + (zsm - z2)*(zsm - z2) )/l;
          double R_2 = 1.0 - R_1;

          // begin contact iteration enforcement
          int N = 5; // number of contact enforcement iterations -- soon to be a parameter
          for (std::size_t n=0;n<N;n++) {
            double alpha1 = 1.0/sqrt(N-(n+1)+1); // Fortran index difference from Eqn. 12
          }
          return slave;
        }

}}}

