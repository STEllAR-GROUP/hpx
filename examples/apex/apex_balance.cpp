////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Oregon University
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <apex_api.hpp>

#include <iostream>
#include <random>
#include <cmath>
#include <list>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
double do_work(boost::uint64_t n);

HPX_PLAIN_ACTION(do_work, do_work_action);

double do_work(boost::uint64_t n) {
    double result = 1;
    for(boost::uint64_t i = 0; i < n; ++i) {
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

int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    boost::uint64_t units = vm["units"].as<boost::uint64_t>();
    boost::uint64_t blocks = vm["blocks"].as<boost::uint64_t>();

    // Keep track of the time required to execute.
    hpx::util::high_resolution_timer t;

    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    std::vector<double> probabilities(localities.size());
    probabilities[0] = 1.0;

    std::cout << "There are " << localities.size() << " localities." << std::endl;
    std::cout << "Units: " << units << " n: " << n << std::endl;

    for(boost::uint64_t block = 0; block < blocks; ++block) {
        std::cout << "Block " << block << std::endl;
        std::list<boost::uint64_t> work(units, n);
        std::list<hpx::lcos::future<double> > futures;
        for(boost::uint64_t & item : work) {
            do_work_action act;
            size_t next = next_locality(probabilities);
            std::cout << "Will issue work to loc " << next << std::endl;
            futures.push_back(hpx::async(act, localities[next], item));
        }
        std::cout << "Issued work for block " << block << std::endl;
        hpx::lcos::wait_all(futures.begin(), futures.end());
        std::cout << "Work done for block " << block << std::endl;
    }


    char const* fmt = "elapsed time: %1% [s]\n";
    std::cout << (boost::format(fmt) % t.elapsed());

    return hpx::finalize(); // Handles HPX shutdown
}

void register_policy(void) {
    apex::register_periodic_policy(1000000, [](apex_context const& context) {
        std::cout << "Periodic policy!" << std::endl;
        return APEX_NOERROR;
    });
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          boost::program_options::value<boost::uint64_t>()->default_value(1000000),
          "n value for do_work")
        ;

    desc_commandline.add_options()
        ( "units",
          boost::program_options::value<boost::uint64_t>()->default_value(100),
          "units of work per block")
        ;

    desc_commandline.add_options()
        ( "blocks",
          boost::program_options::value<boost::uint64_t>()->default_value(10),
          "blocks before program completion")
        ;
    hpx::register_startup_function(register_policy);

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
