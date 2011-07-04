//  Copyright (c) 2011 Hartmut Kaiser
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

#include "contact/contact.hpp"

inline void 
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::components::contact>& accu)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu.push_back(hpx::components::contact(id));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t num_vertices = 0;
    std::size_t iterations = 0;

    if (vm.count("num-vertices"))
        num_vertices = vm["num-vertices"].as<std::size_t>();

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
                hpx::components::contact::server_component_type>();

        hpx::components::distributing_factory::result_type mem_blocks = 
            factory.create_components(mem_block_type, num_vertices);

        std::vector<hpx::components::contact> accu;

        init(locality_results(mem_blocks), accu);

        // initialize the system
        for (std::size_t i=0;i<num_vertices;i++) {
          accu[i].init(i); 
        }

        std::vector<hpx::lcos::future_value<void> > barrier;

        for (std::size_t i=0;i<num_vertices;i++) {
          barrier.push_back(accu[i].contactsearch_async());
        }

        hpx::components::wait(barrier);

        std::vector<hpx::lcos::future_value<void> > barrier2;
        for (std::size_t i=0;i<num_vertices;i++) {
          barrier2.push_back(accu[i].contactenforce_async()); 
        }

        hpx::components::wait(barrier2);
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
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("num-vertices", value<std::size_t>()->default_value(8), 
            "the number of vertices this contact problem works over")
        ("iterations", value<std::size_t>()->default_value(1), 
            "the number of contact enforcement iterations") 
        ;
    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

