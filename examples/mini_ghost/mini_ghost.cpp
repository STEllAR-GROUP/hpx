//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>

#include <examples/mini_ghost/profiling.hpp>
#include <examples/mini_ghost/barrier.hpp>
#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/stepper.hpp>

typedef mini_ghost::grid<double> grid_type;

typedef mini_ghost::stepper<grid_type::value_type> stepper_type;

// Global configuration data (after initialization this is read-only)
mini_ghost::params<grid_type::value_type> p;

// Global profiling data
hpx::lcos::local::spinlock profiling_data_mtx;
boost::shared_ptr<hpx::lcos::local::counting_semaphore> profiling_data_sem;
std::vector<mini_ghost::profiling::profiling_data> profiling_data;

double init_start = 0;

void add_profile(mini_ghost::profiling::profiling_data const & pd)
{
    {
        hpx::lcos::local::spinlock::scoped_lock l(profiling_data_mtx);
        profiling_data.push_back(pd);
        profiling_data_sem->signal();
    }

    if (p.rank == 0)
    {
        profiling_data_sem->wait();
    }
}

HPX_PLAIN_ACTION(add_profile);

int hpx_main(boost::program_options::variables_map& vm)
{
    mini_ghost::profiling::data().time_init(
        hpx::util::high_resolution_timer::now() - init_start);

    hpx::util::high_resolution_timer timer_all;

    hpx::id_type here = hpx::find_here();
    std::string name = hpx::get_locality_name();

    p.rank = hpx::naming::get_locality_id_from_id(here);
    if(p.rank == 0)
    {
        std::cout << "mini ghost started up in "
                  << hpx::util::high_resolution_timer::now() - init_start
                  << " seconds.\n";
    }

    p.nranks = hpx::get_num_localities_sync();

    profiling_data_sem.reset(new hpx::lcos::local::counting_semaphore(p.nranks));
    p.setup(vm);

    // Create the local stepper object, retrieve the local pointer to it
    hpx::id_type stepper_id = hpx::components::new_<stepper_type>(here).get();
    boost::shared_ptr<stepper_type> stepper(
        hpx::get_ptr<stepper_type>(stepper_id).get());

    // Initialize stepper
    stepper->init(p).get();
    mini_ghost::barrier_wait();

    // Perform the actual simulation work
    stepper->run(p.num_spikes, p.num_tsteps);
    mini_ghost::barrier_wait();

    // Output various pieces of information about the run
    if (stepper->get_rank() == 0)
    {
        // Output various pieces of information about the run
        add_profile(mini_ghost::profiling::data());

        if (p.report_perf)
            mini_ghost::profiling::report(std::cout, profiling_data, p);
        else
            std::cout << "Total runtime: " << timer_all.elapsed() << "\n";

        std::ofstream fs("results.yaml");
        mini_ghost::profiling::report(fs, profiling_data, p);
        std::cout << "finalizing ...\n";

        return hpx::finalize();
    }
    else
    {
        // Send performance data from this locality to root
        hpx::apply(add_profile_action(), hpx::find_root_locality(),
            mini_ghost::profiling::data());
        return 0;
    }
}

int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    p.cmd_options(desc_commandline);

    // Register startup functions for creating/retrieving our global barrier
    hpx::register_pre_startup_function(&mini_ghost::create_barrier);
    hpx::register_startup_function(&mini_ghost::find_barrier);

    // Initialize and run HPX such that hpx_main is executed on all localities
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    init_start = hpx::util::high_resolution_timer::now();
    return hpx::init(desc_commandline, argc, argv, cfg);
}
