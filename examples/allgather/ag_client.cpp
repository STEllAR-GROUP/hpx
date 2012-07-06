//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include "ag/point.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

/// This function initializes a vector of \a ag::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<ag::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(ag::point(id));
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

        std::cout << " Number of components: " << np << std::endl;

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<ag::server::point>();

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
        std::vector<ag::point> points;

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

        double ctime = 0.0;

        // get the gids from the components
        std::vector<hpx::naming::id_type> point_components;
        for (std::size_t i=0;i<np;i++) {
          point_components.push_back(points[i].get_gid());
        }
  
        hpx::util::high_resolution_timer computetime;
        {
          std::vector<hpx::lcos::future<void> > compute_phase;
          for (std::size_t i=0;i<np;i++) {
            compute_phase.push_back(points[i].compute_async(point_components));
          }
          hpx::lcos::wait(compute_phase);
        }
        ctime += computetime.elapsed();

        // print out the sums
        for (std::size_t i=0;i<np;i++) {
          points[i].print();
        }

        std::cout << " init time: " << k1time << std::endl;
        std::cout << " compute time: " << ctime << std::endl;

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
            "the number of components");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

