//  Copyright (c)      2011 Hartmut Kaiser
//  Copyright (c)      2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <time.h>

#include "random_mem_access/random_mem_access.hpp"

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    {
        // get list of all known localities
        //std::vector<hpx::naming::id_type> prefixes;
        //hpx::applier::applier& appl = hpx::applier::get_applier();
        //hpx::naming::id_type prefix;

        // create a distributing factory locally
        components::distributing_factory factory;
        factory.create(hpx::applier::get_applier().get_runtime_support_gid());

        component_type function_type = get_component_type<hpx::components::random_mem_access>();

        int array_size = 6;
        result_type functions = factory.create_components(function_type, array_size);

        //if (appl.get_remote_prefixes(prefixes))
        //    // create random_mem_access on any of the remote localities
        //    prefix = prefixes[0];
        //else
        //    // create an accumulator locally
        //    prefix = appl.get_runtime_support_gid();

        //std::vector< hpx::components::random_mem_access > accu;

        //int array_size = 6;
        //accu.resize(array_size);

        //for (int i=0;i<array_size;i++) {
        //  accu[i].create(prefix);
        //}

        init(locality_results(functions), numsteps);

        // initialize the array
        for (int i=0;i<array_size;i++) {
          accu[i].init(i); 
        }

        srand( time(NULL) );

        int N = 10;
        std::vector<hpx::lcos::future_value<void> > barrier;
        for (int i=0;i<N;i++) {
          int rn = rand() % array_size;
          std::cout << " Random element access: " << rn << std::endl;
          barrier.push_back(accu[rn].add_async());
        }

        hpx::components::wait(barrier);

        std::vector<hpx::lcos::future_value<void> > barrier2;
        for (int i=0;i<array_size;i++) {
          barrier2.push_back(accu[i].print_async()); 
        }

        hpx::components::wait(barrier2);
    }

    // initiate shutdown of the runtime systems on all localities
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

