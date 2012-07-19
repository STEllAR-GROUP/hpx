//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>
#include "fname.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "gtc_hpx/server/point.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

extern "C" {void FNAME(gtc)(); }

bool fexists(std::string const filename)
{
  std::ifstream ifile(filename);
  return ifile ? true : false;
}

/// This function initializes a vector of \a gtc::point clients,
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

        ///////////////////////////////////////////////////////////////////////
        // Retrieve the command line options.
        std::string const parfilename = vm["file"].as<std::string>();

        // Get the component type for our point component.
        hpx::components::component_type block_type =
        hpx::components::get_component_type<gtc::server::point>();
        
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());
        
        std::size_t num_partitions = 10;
        hpx::components::distributing_factory::result_type blocks =
                                  factory.create_components(block_type, num_partitions);
        
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<hpx::naming::id_type> components;
        // Populate the client vectors.
        init(hpx::util::locality_results(blocks), components);

        {
          std::vector<hpx::lcos::future<void> > setup_phase;
          gtc::server::point::setup_action setup;
          for (std::size_t i=0;i<num_partitions;i++) {
            setup_phase.push_back(hpx::async(setup,components[i],num_partitions,i,components));
          }
          hpx::lcos::wait(setup_phase);
        }

      //  {
      //    std::vector<hpx::lcos::future<void> > allreduce_phase;
      //    gtc::server::point::allreduce_action allreduce;
      //    for (std::size_t i=0;i<num_partitions;i++) {
      //      allreduce_phase.push_back(hpx::async(allreduce,components[i]));
      //    }
      //    hpx::lcos::wait(allreduce_phase);
      //  }

        {
          std::vector<hpx::lcos::future<void> > chargei_phase;
          gtc::server::point::chargei_action chargei;
          for (std::size_t i=0;i<num_partitions;i++) {
            chargei_phase.push_back(hpx::async(chargei,components[i]));
          }
          hpx::lcos::wait(chargei_phase);
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
        ("file", value<std::string>()->default_value(
                "exp_flush.dat"));

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

