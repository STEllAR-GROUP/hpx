//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007-2011 Matthew Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point_geometry/point.hpp"
#include <boost/geometry/geometries/polygon.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

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

        std::size_t array_size = 8;

        namespace bg = boost::geometry;

        // create a distributing factory locally
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        hpx::components::component_type block_type =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, array_size);

        std::vector<hpx::geometry::point> accu;

        // SIMPLE PROBLEM
        // create some boxes to smash together
        const std::size_t num_bodies = 5;
        const std::size_t numpoints = 5;
        double bbox[num_bodies][4];
        std::size_t imax = 1000;

        srand( time(NULL) );
        std::size_t i = 0;
        while (i < num_bodies) {
          std::size_t rnx = rand() % imax;
          std::size_t rny = rand() % imax;
          std::size_t rn_extent1 = rand() % imax;
          std::size_t rn_extent2 = rand() % imax;

          double xmin = 0.001*(rnx - rn_extent1);
          double ymin = 0.001*(rny - rn_extent2);
          double xmax = 0.001*(rnx + rn_extent1);
          double ymax = 0.001*(rny + rn_extent2);

          bool found = 0;
          for (std::size_t j=0;j<i;j++) {
            if ( intersection(xmin,xmax,ymin,ymax,
              bbox[j][0],bbox[j][1],bbox[j][2],bbox[j][3]) ) {
              found = 1;
              break;
            } 
          }
 
          if ( xmin > 1000 || ymin > 1000 || xmax > 1000 || ymax > 1000 ) {
            found = 1;
          }

          if ( found == 0 ) { 
            bbox[i][0] = xmin;
            bbox[i][1] = xmax;
            bbox[i][2] = ymin;
            bbox[i][3] = ymax;
            i++;
          }
        } 

        // TEST
        //for (i=0;i<num_bodies;i++) {
        //  std::cout << " Bounding box : " << bbox[i][0] << " " << bbox[i][1] << " " << bbox[i][2] << " " << bbox[i][3] << std::endl;
        //} 

        init(locality_results(blocks), accu);


        // Initial Data -----------------------------------------
        std::vector<hpx::lcos::future_value<void> > initial_phase;

        for (i=0;i<num_bodies;i++) {
          initial_phase.push_back(accu[i].init_async(bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3],numpoints));
        }

        hpx::components::wait(initial_phase);

        // Search for Contact------------------------------------

        // vector of futures
        std::vector<hpx::lcos::future_value<bool> > search_phase;

        // vector of gids
        std::vector<hpx::naming::id_type> search_objects;
        for (i=0;i<num_bodies;i++) {
          search_objects.resize(num_bodies-(i+1));
          for (std::size_t j=i+1;j<num_bodies;j++) {
            search_objects[j-(i+1)] = accu[j].get_gid();
          }
          search_phase.push_back(accu[i].search_async(search_objects));
        }

        hpx::components::wait(search_phase);

        // Contact enforcement ----------------------------------

        // Move bodies--------------------------------------------
        std::vector<hpx::lcos::future_value<void> > move_phase;
        for (i=0;i<num_bodies;i++) {
          move_phase.push_back(accu[i].move_async());
        }
        hpx::components::wait(move_phase);

#if 0
        hpx::geometry::point pt5(hpx::find_here(), 0.5, 0.5);
        bool inside = bg::within(pt5, p);
        std::cout << "Point is " << (inside ? "inside" : "outside") << std::endl;
#endif
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("point_geometry_client", argc, argv); // Initialize and run HPX
}
