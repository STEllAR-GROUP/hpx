//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <examples/mini_ghost/barrier.hpp>
#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/stepper.hpp>

typedef mini_ghost::grid<double> grid_type;

typedef mini_ghost::stepper<grid_type::value_type> stepper_type;

mini_ghost::params<grid_type::value_type> p;

int hpx_main(boost::program_options::variables_map& vm)
{
    hpx::util::high_resolution_timer timer_all;
    hpx::id_type    here = hpx::find_here();
    std::string     name = hpx::get_locality_name();
    p.rank = hpx::naming::get_locality_id_from_id(here);
    p.nranks = hpx::get_num_localities().get();

    p.setup(vm);

    hpx::id_type stepper_id
        = hpx::components::new_<stepper_type>(hpx::find_here()).get();

    boost::shared_ptr<stepper_type>
        stepper(hpx::get_ptr<stepper_type>(stepper_id).get());

    stepper->init(p);

    stepper->run(p.num_spikes, p.num_tsteps);

    mini_ghost::barrier_wait();
    std::cout << "Total runtime: " << timer_all.elapsed() << "\n";
    if (p.rank==0)
      return hpx::finalize();
    else return 0;
}

int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    p.cmd_options(desc_commandline);

    // Register startup functions for creating our global barrier
    hpx::register_pre_startup_function(&mini_ghost::create_barrier);
    hpx::register_startup_function(&mini_ghost::find_barrier);

    // Initialize and run HPX, this test requires to run hpx_main on all localities
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}
