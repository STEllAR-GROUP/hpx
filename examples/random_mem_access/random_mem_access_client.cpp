//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/foreach.hpp>

#include <cstddef>
#include <vector>
#include <time.h>

#include "random_mem_access/random_mem_access.hpp"

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t array_size = 0;
    std::size_t iterations = 0;

    if (vm.count("array-size"))
        array_size = vm["array-size"].as<std::size_t>();

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    {
        std::vector<hpx::components::random_mem_access> accu =
            hpx::new_<hpx::components::random_mem_access[]>(
                hpx::default_layout(hpx::find_all_localities()),
                    array_size).get();

        // initialize the array
        for (std::size_t i=0;i<array_size;i++) {
          accu[i].init(i);
        }

        srand( time(nullptr) );

        std::vector<hpx::lcos::future<void> > barrier;
        for (std::size_t i=0;i<iterations;i++) {
          std::size_t rn = rand() % array_size;
          //std::cout << " Random element access: " << rn << std::endl;
          barrier.push_back(accu[rn].add_async());
        }

        hpx::wait_all(barrier);

        std::vector<hpx::lcos::future<void> > barrier2;
        for (std::size_t i=0;i<array_size;i++) {
          barrier2.push_back(accu[i].print_async());
        }

        hpx::wait_all(barrier2);
    }

    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::value;

    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("array-size", value<std::size_t>()->default_value(8),
            "the size of the array")
        ("iterations", value<std::size_t>()->default_value(16),
            "the number of lookups to perform")
        ;
    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

