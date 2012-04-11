//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/future_wait.hpp>

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
          return 0;
#if 0
            typedef std::vector<lcos::future<polygon_type> > lazy_results_type;

            lazy_results_type lazy_results;
            BOOST_FOREACH(naming::id_type gid, search_objects)
            {
              lazy_results.push_back( stubs::point::get_poly_async( gid ) );
            }

            // will return the number of invoked futures
            bool redo = false;
            lcos::wait(lazy_results, boost::bind(&point::search_callback, this, _1, _2,boost::ref(redo)));
#endif
            if ( slave_.size() > 0 ) return 1;
            else return 0;
        }

        bool point::search_callback(std::size_t i, polygon_type const& poly,bool &redo)
        {

          double tmp = 0;
          for (std::size_t k=0;k<1000;k++) {
            tmp += rand();
          }
          //std::cout << " TEST in callback " << i << " object id " << objectid_ << std::endl;

          // This contains the R1/R2 sums needed to compute alpha2
          R_.resize(poly_.outer().size());
          // initialize
          for (std::size_t j=0;j<poly_.outer().size();j++) {
            R_[j] = tmp;
          }

          if ( i != objectid_ ) {
            // search for contact
            typedef boost::geometry::model::linestring<point_type> linestring_type;

            // Check for contact
            std::deque<polygon_type> output;
            boost::geometry::intersection(poly_,poly,output);
            BOOST_FOREACH(polygon_type const& p, output) {
              linestring_type line;
              line.resize(2);

              //char basename[80];
              //sprintf(basename,"overlap%d_%d.dat",(int) objectid_,(int) i);
              //std::ofstream file;
              //file.open(basename);
              // note that the first and the last vertices are the same
              for (std::size_t j=0;j<p.outer().size()-1;j++) {
               // file << (p.outer())[j].x() << " " << (p.outer())[j].y() << std::endl;

                point_type const& pp = (p.outer())[j];

                // find the vertex which belongs to poly_
                // record the point as a slave
                // find the index of poly_ that corresponds to this point
                polygon_type::ring_type const& outer_ = poly_.outer();
                polygon_type::ring_type const& outer = poly.outer();
                bool found_ = false;
                bool found = false;
                for (std::size_t k=0;k<poly_.outer().size();k++) {
                  if ( boost::geometry::distance(pp,outer_[k]) < 1.e-10 ) {
                    slave_.push_back(k);
                    found_ = true;
                    break;
                  }
                }
                if ( !found_ ) {
                  // it may belong to poly
                  for (std::size_t k=0;k<poly.outer().size();k++) {
                    if ( boost::geometry::distance(pp,outer[k]) < 1.e-10 ) {
                      inv_slave_.push_back(k);
                      inv_object_id_.push_back(i);
                      found = true;
                      break;
                    }
                  }
                }

                if ( found_ ) {
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

                // find the inverse
                if ( found ) {
                  // The following section will be replaced by Barend
                  // with the nearest neighbor routine
                  // but for now, this works

                  // the master segment should belong to poly
                  double mindist = 999.;
                  int min_k = -1;
                  int final;
                  for (std::size_t k=0;k<poly_.outer().size();k++) {
                    final = k+1;
                    if ( k+1 >= poly_.outer().size() ) final = 0;
                    line[0] = (poly_.outer())[k];
                    line[1] = (poly_.outer())[final];
                    double testdist = boost::geometry::distance(pp,line);

                    if ( testdist < mindist ) {
                      mindist = testdist;
                      min_k = k;
                    }
                  }

                  BOOST_ASSERT(min_k >= 0 );
                  inv_master_.push_back(min_k);
                }

              }
              //file.close();
            }

            BOOST_ASSERT(slave_.size() == master_.size());
            BOOST_ASSERT(slave_.size() == object_id_.size());

            // initialize the change in velocities
            change_vx_.resize( slave_.size() );
            change_vy_.resize( slave_.size() );
            for (std::size_t j=0;j<slave_.size();j++) {
              change_vx_[j] = 0.0;
              change_vy_[j] = 0.0;
            }


            BOOST_ASSERT(inv_slave_.size() == inv_master_.size());
            BOOST_ASSERT(inv_slave_.size() == inv_object_id_.size());

            for (std::size_t j=0;j<inv_slave_.size();j++) {
              std::size_t master_vertex = inv_master_[j];
              std::size_t final = master_vertex + 1;
              if ( final >= poly_.outer().size() ) final = 0;

              double x1 = (poly_.outer())[master_vertex].x();
              double x2 = (poly_.outer())[final].x();
              double z1 = (poly_.outer())[master_vertex].y();
              double z2 = (poly_.outer())[final].y();

              double l = boost::geometry::distance((poly_.outer())[master_vertex],(poly_.outer())[final]);
              double A = (z2-z1)/l;
              double B = (x1-x2)/l;
              double C = (x2*z1-x1*z2)/l;

              double xs = (poly.outer())[inv_slave_[j]].x();
              double zs = (poly.outer())[inv_slave_[j]].y();
              double delta = -(A*xs + B*zs + C);

              double xsm = xs + A*delta;
              double zsm = zs + B*delta;

              double D1 = sqrt( (xsm-x1)*(xsm-x1) + (zsm - z1)*(zsm - z1) );
              double D2 = sqrt( (xsm-x2)*(xsm-x2) + (zsm - z2)*(zsm - z2) );
              double D3 = D1 + D2;

              R_[master_vertex] += D2/D3;
              R_[final] += D1/D3;
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

        void point::move(double dt,double time)
        {
          // clear out slave nodes and master segments
          slave_.resize(0);
          master_.resize(0);
          object_id_.resize(0);
          inv_slave_.resize(0);
          inv_master_.resize(0);
          inv_object_id_.resize(0);
          for (std::size_t i=0;i<poly_.outer().size();i++) {
            point_type &p = (poly_.outer())[i];
            (poly_.outer())[i].x(p.x() + velx_[i]*dt);
            (poly_.outer())[i].y(p.y() + vely_[i]*dt);
          }

          char basename[80];
          sprintf(basename,"geo%d.dat",(int) objectid_);
          std::ofstream file;
          if ( time < 1.e-8 ) {
            file.open(basename);
          } else {
            file.open(basename,std::ios::app);
            file << " " << std::endl << std::endl;
          }
          for (std::size_t i=0;i<poly_.outer().size();i++) {
            file << (poly_.outer())[i].x() << " " << (poly_.outer())[i].y() << std::endl;
          }
          file.close();
        }

        void point::enforce(std::vector<hpx::naming::id_type> const& master_gids,double dt,std::size_t n,std::size_t N)
        {
          typedef std::vector<lcos::future<polygon_type> > lazy_results_type;

          lazy_results_type lazy_results;

          for (std::size_t i=0;i<slave_.size();i++) {
            naming::id_type gid = master_gids[ object_id_[i] ];
            lazy_results.push_back( stubs::point::get_poly_async( gid ) );
          }

          // will return the number of invoked futures
          lcos::wait(lazy_results, boost::bind(&point::enforce_callback, this, _1, _2,boost::ref(dt),boost::ref(n),boost::ref(N)));
        }

        bool point::enforce_callback(std::size_t i, polygon_type const& poly,double dt,std::size_t n,std::size_t N)
        {

          std::size_t master_vertex = master_[i];
          std::size_t final = master_[i] + 1;
          if ( final >= poly.outer().size() ) final = 0;

          // Masses -- equal for now
          double M_s = 1.0;
          double M_1 = 1.0;
          double M_2 = 1.0;

          // the slave node
          point_type const& pp = (poly_.outer())[slave_[i]];

          double l = boost::geometry::distance((poly.outer())[master_vertex],(poly.outer())[final]);

          // For debugging
          //typedef boost::geometry::model::linestring<point_type> linestring_type;
          //linestring_type line;
          //line.resize(2);
          //line[0] = (poly.outer())[master_vertex];
          //line[1] = (poly.outer())[final];
          //double tdelta = boost::geometry::distance(pp,line);

          double x1 = (poly.outer())[master_vertex].x();
          double x2 = (poly.outer())[final].x();
          double z1 = (poly.outer())[master_vertex].y();
          double z2 = (poly.outer())[final].y();

          double A = (z2-z1)/l;
          double B = (x1-x2)/l;
          double C = (x2*z1-x1*z2)/l;

          double xs = pp.x();
          double zs = pp.y();
          double delta = -(A*xs + B*zs + C);

          double xsm = xs + A*delta;
          double zsm = zs + B*delta;

          double R_1 = sqrt( (xsm-x2)*(xsm-x2) + (zsm - z2)*(zsm - z2) )/l;
          double R_2 = 1.0 - R_1;

          // Here we assume a slave never has more than one master segment
          // I am not certain what is meant by sum(R_m) in Eqn 13-- seems to be 1 all the time
          // Need to check with Johnson or somebody who knows
          double RM = 1.0;
          double alpha2 = RM/(RM + R_[slave_[i]]);

          double alpha1 = 1.0/sqrt(N-(n+1.0)+1.0); // Fortran index difference from Eqn. 12
          double alpha  = alpha1*alpha2;

          // Eqn 10
          double dv = -alpha*delta/dt/(1. + pow(R_1,2)*M_s/M_1 + pow(R_2,2)*M_s/M_2);

          // Eqn 14-15
          change_vx_[ i ] += -A*dv;
          change_vy_[ i ] += -B*dv;

          // return type says continue or not
          // usually return true
          return true;
        }

        void point::adjust(double dt)
        {
          std::size_t duplicate = (poly_.outer()).size()-1;
          for (std::size_t i=0;i<slave_.size();i++) {
            point_type &p = (poly_.outer())[ slave_[i] ];
            (poly_.outer())[ slave_[i] ].x(p.x() + change_vx_[i]*dt);
            (poly_.outer())[ slave_[i] ].y(p.y() + change_vy_[i]*dt);

            velx_[ slave_[i] ] += change_vx_[i];
            vely_[ slave_[i] ] += change_vy_[i];

            // Take care of the duplicate node
            if ( slave_[i] == 0 ) {
              (poly_.outer())[duplicate].x(p.x());
              (poly_.outer())[duplicate].y(p.y());
              velx_[ duplicate ] = velx_[ slave_[i] ];
              vely_[ duplicate ] = vely_[ slave_[i] ];
            }

            // reset the change to zero
            change_vx_[ i ] = 0.0;
            change_vy_[ i ] = 0.0;
          }
        }

        /// Recompute Rsum
        void point::recompute(std::vector<hpx::naming::id_type> const& search_objects)
        {
          typedef std::vector<lcos::future<polygon_type> > lazy_results_type;

          lazy_results_type lazy_results;

          // Re-initialize Rsum
          for (std::size_t i=0;i<R_.size();i++) {
            R_[i] = 0.0;
          }

          for (std::size_t i=0;i<inv_slave_.size();i++) {
            naming::id_type gid = search_objects[ inv_object_id_[i] ];
            lazy_results.push_back( stubs::point::get_poly_async( gid ) );
          }

          // will return the number of invoked futures
          lcos::wait(lazy_results, boost::bind(&point::recompute_callback, this, _1, _2));
        }

        bool point::recompute_callback(std::size_t i, polygon_type const& poly)
        {
          std::size_t master_vertex = inv_master_[i];
          std::size_t final = master_vertex + 1;
          if ( final >= poly_.outer().size() ) final = 0;

          double x1 = (poly_.outer())[master_vertex].x();
          double x2 = (poly_.outer())[final].x();
          double z1 = (poly_.outer())[master_vertex].y();
          double z2 = (poly_.outer())[final].y();

          double l = boost::geometry::distance((poly_.outer())[master_vertex],(poly_.outer())[final]);
          double A = (z2-z1)/l;
          double B = (x1-x2)/l;
          double C = (x2*z1-x1*z2)/l;

          double xs = (poly.outer())[inv_slave_[i]].x();
          double zs = (poly.outer())[inv_slave_[i]].y();
          double delta = -(A*xs + B*zs + C);

          double xsm = xs + A*delta;
          double zsm = zs + B*delta;

          double D1 = sqrt( (xsm-x1)*(xsm-x1) + (zsm - z1)*(zsm - z1) );
          double D2 = sqrt( (xsm-x2)*(xsm-x2) + (zsm - z2)*(zsm - z2) );
          double D3 = D1 + D2;

          R_[master_vertex] += D2/D3;
          R_[final] += D1/D3;

          // return type says continue or not
          // usually return true
          return true;
        }

}}}

