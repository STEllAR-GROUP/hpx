//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include "fname.h"
#include "ep/point.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

// extern "C" {void FNAME(hello)(); }

/// This function initializes a vector of \a ep::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<ep::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(ep::point(id));
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
        std::size_t const number_partitions = vm["number_partitions"].as<std::size_t>();
        std::size_t scale = vm["scale"].as<std::size_t>();
        scale *= 1E8;

        // number_partitions defines the size of the partition
        // for Additive Schwarz to work, we will need more partitions
        // than just number_partitions.  number_partitions should be as
        // small as possible for performance reasons; however, it can't be too
        // small since the partitioned graph won't fit into memory if it is too small
        std::size_t num_pe = number_partitions; // actual number of partitions is num_pe
        std::cout << " Number of components: " << num_pe << std::endl;

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<ep::server::point>();

        // ---------------------------------------------------------------
        // Create ne point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, num_pe);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<ep::point> points;

        // Populate the client vectors.
        init(hpx::util::locality_results(blocks), points);

        // Begin Kernel 2
        hpx::util::high_resolution_timer kernel2time;
        {
          std::vector<hpx::lcos::future<void> > bfs_phase;
          for (std::size_t i=0;i<num_pe;i++) {
            bfs_phase.push_back(points[i].bfs_async(scale));
          }
          hpx::lcos::wait(bfs_phase);
        }
        double k2time = kernel2time.elapsed();

        std::cout << " Time: " << k2time << std::endl;

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

//     FNAME(hello)();

    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("number_partitions", value<std::size_t>()->default_value(2),
            "the number of components")
        ("scale", value<std::size_t>()->default_value(1000),
            "the scale of the problem");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

