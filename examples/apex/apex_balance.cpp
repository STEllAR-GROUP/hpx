////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Oregon University
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <apex_api.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <list>
#include <random>
#include <vector>

// our apex policy handle
apex_policy_handle * periodic_policy_handle;

double do_work(std::uint64_t n);

HPX_PLAIN_ACTION(do_work, do_work_action);

double do_work(std::uint64_t n) {
    double result = 1;
    for(std::uint64_t i = 0; i < n; ++i) {
        result += std::sin((double)i) * (double)i;
    }
    return result;
}


size_t next_locality( const std::vector<double> & probs) {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double eps = 1e-9;
    double r = distribution(generator);

    size_t i = 0;
    for(const double p : probs) {
        r -= p;
        if(r < eps) {
            return i;
        }
        ++i;
    }
    return 0;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();
    std::uint64_t units = vm["units"].as<std::uint64_t>();
    std::uint64_t blocks = vm["blocks"].as<std::uint64_t>();

    // Keep track of the time required to execute.
    hpx::chrono::high_resolution_timer t;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    std::vector<double> probabilities(localities.size());
    probabilities[0] = 1.0;

    std::cout << "There are " << localities.size() << " localities." << std::endl;
    std::cout << "Units: " << units << " n: " << n << std::endl;

    for(std::uint64_t block = 0; block < blocks; ++block) {
        std::cout << "Block " << block << std::endl;
        std::list<std::uint64_t> work(units, n);
        std::list<hpx::lcos::future<double> > futures;
        for(std::uint64_t & item : work) {
            do_work_action act;
            size_t next = next_locality(probabilities);
            std::cout << "Will issue work to loc " << next << std::endl;
            futures.push_back(hpx::async(act, localities[next], item));
        }
        std::cout << "Issued work for block " << block << std::endl;
        hpx::lcos::wait_all(futures.begin(), futures.end());
        std::cout << "Work done for block " << block << std::endl;
    }


    char const* fmt = "elapsed time: {1} [s]\n";
    hpx::util::format_to(std::cout, fmt, t.elapsed());

    apex::deregister_policy(periodic_policy_handle);

    return hpx::finalize(); // Handles HPX shutdown
}

void register_policy(void) {
    periodic_policy_handle =
        apex::register_periodic_policy(1000000, [](apex_context const&) {
            std::cout << "Periodic policy!" << std::endl;
            return APEX_NOERROR;
        });
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    hpx::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          hpx::program_options::value<std::uint64_t>()->default_value(1000000),
          "n value for do_work")
        ;

    desc_commandline.add_options()
        ( "units",
          hpx::program_options::value<std::uint64_t>()->default_value(100),
          "units of work per block")
        ;

    desc_commandline.add_options()
        ( "blocks",
          hpx::program_options::value<std::uint64_t>()->default_value(10),
          "blocks before program completion")
        ;
    hpx::register_startup_function(register_policy);

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
