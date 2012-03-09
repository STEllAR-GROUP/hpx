//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>
#include <time.h>

#include "random_mem_access/random_mem_access.hpp"

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::components::random_mem_access>& accu)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu.push_back(hpx::components::random_mem_access(id));
    }
}

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
        // get list of all known localities
        //std::vector<hpx::naming::id_type> prefixes;
        //hpx::applier::applier& appl = hpx::applier::get_applier();
        //hpx::naming::id_type prefix;

        // create a distributing factory locally
        hpx::components::distributing_factory factory;
        factory.create(hpx::applier::get_applier().get_runtime_support_gid());

        hpx::components::component_type mem_block_type =
            hpx::components::get_component_type<
                hpx::components::random_mem_access::server_component_type>();

        hpx::components::distributing_factory::result_type mem_blocks =
            factory.create_components(mem_block_type, array_size);

        //if (appl.get_remote_prefixes(prefixes))
        //    // create random_mem_access on any of the remote localities
        //    prefix = prefixes[0];
        //else
        //    // create an accumulator locally
        //    prefix = appl.get_runtime_support_gid();

        std::vector<hpx::components::random_mem_access> accu;

        //int array_size = 6;
        //accu.resize(array_size);

        //for (int i=0;i<array_size;i++) {
        //  accu[i].create(prefix);
        //}

        ::init(locality_results(mem_blocks), accu);

        // initialize the array
        for (std::size_t i=0;i<array_size;i++) {
          accu[i].init(i);
        }

        srand( time(NULL) );

        std::vector<hpx::lcos::future<void> > barrier;
        for (std::size_t i=0;i<iterations;i++) {
          std::size_t rn = rand() % array_size;
          //std::cout << " Random element access: " << rn << std::endl;
          barrier.push_back(accu[rn].add_async());
        }

        hpx::lcos::wait(barrier);

        std::vector<hpx::lcos::future<void> > barrier2;
        for (std::size_t i=0;i<array_size;i++) {
          barrier2.push_back(accu[i].print_async());
        }

        hpx::lcos::wait(barrier2);
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

