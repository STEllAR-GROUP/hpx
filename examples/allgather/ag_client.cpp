//  Copyright (c) 2012 Matthew Anderson
//  Copyright (c) 2012-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_all.hpp>

#include "ag/server/allgather.hpp"
#include "ag/server/allgather_and_gate.hpp"

#include <iostream>
#include <vector>
#include <math.h>

///////////////////////////////////////////////////////////////////////
// Create a distributing factory locally. The distributing factory can
// be used to create blocks of components that are distributed across
// all localities that support that component type.
void test_allgather(std::size_t np)
{
    // Get the component type for our point component.
    hpx::components::component_type block_type =
        ag::server::allgather::get_component_type();

    std::vector<hpx::id_type> localities =
        hpx::find_all_localities(block_type);
    std::cout << " Number of localities: " << localities.size()
              << std::endl;

    // Create np allgather components. These components will be evenly
    // distributed among all available localities supporting the
    // component type.
    std::vector<hpx::id_type> components =
        hpx::new_<ag::server::allgather[]>(
            hpx::default_layout(localities), np).get();

    hpx::util::high_resolution_timer kernel1time;
    {
      std::vector<hpx::lcos::future<void> > init_phase;
      ag::server::allgather::init_action init;
      for (std::size_t i=0;i<np;i++) {
        init_phase.push_back(hpx::async(init, components[i], i, np));
      }
      hpx::wait_all(init_phase);
    }
    double k1time = kernel1time.elapsed();

    hpx::util::high_resolution_timer computetime;
    {
      std::vector<hpx::lcos::future<void> > compute_phase;
      ag::server::allgather::compute_action compute;
      for (std::size_t i=0;i<np;i++) {
        compute_phase.push_back(hpx::async(compute, components[i], components));
      }
      hpx::wait_all(compute_phase);
    }
    double ctime = computetime.elapsed();

    // print out the sums
    ag::server::allgather::print_action print;
    for (std::size_t i=0;i<np;i++) {
      print(components[i]);
    }

    std::cout << " init time: " << k1time << std::endl;
    std::cout << " compute time: " << ctime << std::endl;
}

void test_allgather_and_gate(std::size_t np)
{
    // Get the component type for our point component.
    hpx::components::component_type block_type =
        ag::server::allgather_and_gate::get_component_type();

    std::vector<hpx::naming::id_type> localities =
        hpx::find_all_localities(block_type);
    std::cout << " Number of localities: " << localities.size()
              << std::endl;

    // Create np allgather components with distributing factory.
    // These components will be evenly distributed among all available
    // localities supporting the component type.
    std::vector<hpx::id_type> components =
        hpx::new_<ag::server::allgather_and_gate[]>(
            hpx::default_layout(localities), np).get();

    hpx::util::high_resolution_timer inittimer;
    {
      std::vector<hpx::lcos::future<void> > init_phase;
      ag::server::allgather_and_gate::init_action init;
      for (std::size_t i = 0; i < np; ++i)
      {
        init_phase.push_back(hpx::async(init, components[i], components, i));
      }
      hpx::wait_all(init_phase);
    }
    double inittime = inittimer.elapsed();

    hpx::util::high_resolution_timer computetimer;
    {
      std::vector<hpx::lcos::future<void> > compute_phase;
      ag::server::allgather_and_gate::compute_action compute;
      for (std::size_t i = 0; i < np; ++i)
      {
        compute_phase.push_back(hpx::async(compute, components[i], 100));
      }
      hpx::wait_all(compute_phase);
    }
    double computetime = computetimer.elapsed();

    // print out the sums
    ag::server::allgather_and_gate::print_action print;
    for (std::size_t i = 0; i < np; ++i)
    {
      print(components[i]);
    }

    std::cout << " init time: " << inittime << std::endl;
    std::cout << " compute time: " << computetime << std::endl;
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

        test_allgather(np);
        //test_allgather_and_gate(np);

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

