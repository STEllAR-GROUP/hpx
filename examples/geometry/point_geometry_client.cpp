//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point_geometry/point.hpp"
#include <boost/geometry/geometries/polygon.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

using hpx::util::high_resolution_timer;

namespace hpx { namespace geometry
{
    typedef boost::geometry::model::polygon<hpx::geometry::point> polygon_2d;
}}

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::point>& accu)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu.push_back(hpx::geometry::point(id));
    }
}

bool floatcmp_le(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;

  if ( x1 < x2 ) return true;

  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}

bool intersection(double xmin,double xmax,
                  double ymin,double ymax,
                  double xmin2,double xmax2,
                  double ymin2,double ymax2)
{
  double pa[2],ea[2];
  static double const half = 0.5;
  pa[0] = half*(xmax + xmin);
  pa[1] = half*(ymax + ymin);

  ea[0] = xmax - pa[0];
  ea[1] = ymax - pa[1];

  double pb[2],eb[2];
  pb[0] = half*(xmax2 + xmin2);
  pb[1] = half*(ymax2 + ymin2);

  eb[0] = xmax2 - pb[0];
  eb[1] = ymax2 - pb[1];
  double T[3];
  T[0] = pb[0] - pa[0];
  T[1] = pb[1] - pa[1];

  if ( floatcmp_le(fabs(T[0]),ea[0] + eb[0]) &&
       floatcmp_le(fabs(T[1]),ea[1] + eb[1]) ) {
    return true;
  } else {
    return false;
  }

}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
       high_resolution_timer t;

        // create some boxes to smash together
        const std::size_t num_bodies = 16;

        namespace bg = boost::geometry;

        // create a distributing factory locally
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        hpx::components::component_type block_type =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, num_bodies);

        std::vector<hpx::geometry::point> accu;

        // SIMPLE PROBLEM
        // create some boxes to smash together
        const std::size_t numpoints = 80000;

        // NOTE: this puts appx 8k on the stack
        double bbox[num_bodies][4];
        double velx[num_bodies];
        double vely[num_bodies];

        std::size_t i = 0;

        // Object #1
        bbox[0][0] =  -2.0;
        bbox[0][1] =   0.0;
        bbox[0][2] =   0.0;
        bbox[0][3] =   2.0;

        velx[0] = 1.0;
        vely[0] = 0.0;

        // Object #2
        bbox[1][0] =   0.001;
        bbox[1][1] =   1.5;
        bbox[1][2] =   0.75;
        bbox[1][3] =   1.25;

        velx[1] = -1.0;
        vely[1] =  0.0;

        for (i=2;i<num_bodies;i++) {
          bbox[i][0] =  bbox[i-1][0]+2;
          bbox[i][1] =  bbox[i-1][1];
          bbox[i][2] =  bbox[i-1][2];
          bbox[i][3] =  bbox[i-1][3];
          velx[i] = -1.0;
          vely[i] =  0.0;
        }

        // Initialize the data
        ::init(locality_results(blocks), accu);


        // Initial Data -----------------------------------------
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (i=0;i<num_bodies;i++) {
          // compute the initial velocity so that everything heads to the origin
          initial_phase.push_back(accu[i].init_async(bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3],velx[i],vely[i],numpoints,i));
        }

        // vector of gids
        std::vector<hpx::naming::id_type> master_objects;
        for (i=0;i<num_bodies;i++) {
          master_objects.push_back(accu[i].get_gid());
        }

        // We have to wait for the futures to finish before exiting.

        hpx::lcos::wait(initial_phase);

        std::cout << " TEST A " << std::endl;
        const double dt = 0.025; // guess for start dt
        const double stop_time = 0.035;
        double time = 0.0;

        while (time < stop_time) {
            //{
              // Move bodies--------------------------------------------
            //  std::vector<hpx::lcos::promise<void> > move_phase;
            //  for (i=0;i<num_bodies;i++) {
            //    move_phase.push_back(accu[i].move_async(dt,time));
            //  }
            //  hpx::lcos::wait(move_phase);
            //}

            time += dt;

        std::cout << " TEST B " << std::endl;
            std::vector<int> search_vector;
            {
              // Search for Contact------------------------------------
              // vector of futures
              std::vector<hpx::lcos::promise<int> > search_phase;

              for (i=0;i<num_bodies;i++) {
                search_phase.push_back(accu[i].search_async(master_objects));
              }

              hpx::lcos::wait(search_phase,search_vector);
            }
        std::cout << " TEST C " << std::endl;

#if 0
            // Contact enforcement ----------------------------------
            BOOST_ASSERT(search_vector.size() == num_bodies);

            std::size_t N = 10; // number of iterations; soon to be a parameter

            for (std::size_t n=0;n<N;n++) {
              std::vector<hpx::lcos::promise<void> > enforcement_phase;
              for (i=0;i<num_bodies;i++) {
                if ( search_vector[i] == 1 ) {
                  // contact was discovered  -- enforce the contact
                  enforcement_phase.push_back(accu[i].enforce_async(master_objects,dt,n,N));
                }
              }
              hpx::lcos::wait(enforcement_phase);

              std::vector<hpx::lcos::promise<void> > adjustment_phase;
              for (i=0;i<num_bodies;i++) {
                if ( search_vector[i] == 1 ) {
                  // adjust the nodes based on the iteration results
                  adjustment_phase.push_back(accu[i].adjust_async(dt));
                }
              }
              hpx::lcos::wait(enforcement_phase);

              // Recompute the Rsum quantity------------------------------------
              std::vector<hpx::lcos::promise<void> > recompute_phase;

              for (i=0;i<num_bodies;i++) {
                if ( search_vector[i] == 1 ) {
                  recompute_phase.push_back(accu[i].recompute_async(master_objects));
                }
              }

              hpx::lcos::wait(recompute_phase);
            }
#endif
          break;
        } // time loop
//#endif

      std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;
    } // ensure things are go out of scope

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("point_geometry_client", argc, argv); // Initialize and run HPX
}
