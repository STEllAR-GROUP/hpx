//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include "ad/point.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

/// This function initializes a vector of \a ad::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<ad::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(ad::point(id));
    }
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        ///////////////////////////////////////////////////////////////////////
        // Retrieve the command line options.
        std::size_t const np = vm["np"].as<std::size_t>();
        std::size_t const nt = vm["nt"].as<std::size_t>();

        std::cout << " Number of components: " << np << std::endl;
        std::cout << " Number of timesteps: " <<  nt << std::endl;

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<ad::server::point>();

        std::vector<hpx::naming::id_type> localities = hpx::find_all_localities(block_type);
        std::size_t numloc = localities.size();
        std::cout << " Number of localities: " << numloc << std::endl;
        // ---------------------------------------------------------------
        // Create ne point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, np);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<ad::point> points;

        // Populate the client vectors.
        init(hpx::util::locality_results(blocks), points);

        hpx::util::high_resolution_timer kernel1time;
        {
          std::vector<hpx::lcos::future<void> > init_phase;
          for (std::size_t i=0;i<np;i++) {
            init_phase.push_back(points[i].init_async(i,np));
          }
          hpx::lcos::wait(init_phase);
        }
        double k1time = kernel1time.elapsed();

        double gtime = 0.0;
        double ctime = 0.0;
        double rtime = 0.0;
        double regtime = 0.0;

        std::vector<std::size_t> available_components;
        for (std::size_t step=0;step<nt;step++) {
          hpx::util::high_resolution_timer gidtime;
          // get the gids from the components
          std::vector<hpx::naming::id_type> point_components;
          for (std::size_t i=0;i<np;i++) {
            point_components.push_back(points[i].get_gid());
          }
          gtime += gidtime.elapsed();
  
          hpx::util::high_resolution_timer computetime;
          {
            std::vector<hpx::lcos::future<void> > compute_phase;
            for (std::size_t i=0;i<np;i++) {
              compute_phase.push_back(points[i].compute_async(point_components));
            }
            hpx::lcos::wait(compute_phase);
          }
          ctime += computetime.elapsed();

          hpx::util::high_resolution_timer rhstime;
          {
            std::vector<hpx::lcos::future<void> > rhs_phase;
            for (std::size_t i=0;i<np;i++) {
              rhs_phase.push_back(points[i].calcrhs_async());
            }
            hpx::lcos::wait(rhs_phase);
          }
          rtime += rhstime.elapsed();

          // regrid -- remove some components
          //hpx::util::high_resolution_timer regridtime;
          //{
          //  std::vector<hpx::lcos::future<void> > regrid_phase;
          //  for (std::size_t i=0;i<np;i++) {
          //    regrid_phase.push_back(points[i].remove_item_async(step,step+1));
          //  }
          //  hpx::lcos::wait(regrid_phase);
          //}
          //regtime += regridtime.elapsed();
          
        }

        std::cout << " init time: " << k1time << std::endl;
        std::cout << " gid  time: " << gtime << std::endl;
        std::cout << " compute time: " << ctime << std::endl;
        std::cout << " rhs time: " << rtime << std::endl;
        std::cout << " remove time: " << regtime << std::endl;

        // Print the total walltime that the computation took.
        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;
    } // Ensure things go out of scope before hpx::finalize is called.

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::value;

    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("np", value<std::size_t>()->default_value(12),
            "the number of components")
        ("nt", value<std::size_t>()->default_value(10),
            "the number of timesteps");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

