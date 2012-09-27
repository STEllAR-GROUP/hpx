//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <vector>
#include <math.h>
#include "fname.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include "gtcx_hpx/server/partition.hpp"

/// This function initializes a vector of \a gtcx::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<hpx::naming::id_type>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(id);
    }
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        // Retrieve the command line options.
        std::size_t const os_factor = vm["os_factor"].as<std::size_t>();

        // Get the component type for our point component.
        hpx::components::component_type block_type =
        hpx::components::get_component_type<gtcx::server::partition>();

        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        std::size_t num_partitions = os_factor * hpx::find_all_localities(block_type).size();
        BOOST_ASSERT(num_partitions);

        std::cout << "num_partitions = " << num_partitions << "\n";

        hpx::components::distributing_factory::result_type blocks =
                                  factory.create_components(block_type, num_partitions);

        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<hpx::naming::id_type> components;
        // Populate the client vectors.
        init(hpx::util::locality_results(blocks), components);

        {
          std::vector<hpx::lcos::future<void> > loop_phase;
          gtcx::server::partition::loop_action loop;
          for (std::size_t i=0;i<num_partitions;i++) {
            loop_phase.push_back(hpx::async(loop,components[i],num_partitions,i,components));
          }
          hpx::lcos::wait(loop_phase);
        }
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
        ("os_factor", value<std::size_t>()->default_value(4),
            "the oversubscription factor in the number of components");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

