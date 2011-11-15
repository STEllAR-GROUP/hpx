//  Copyright (c) 2011 Hartmut Kaiser
//  Copyright (c) 2011 Matt Anderson
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/include/iostreams.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>

#include <cstdlib>
#include <ctime>

#include "element/element.hpp"

/// This function initializes a vector of \a random_mem_access::element clients, 
/// connecting them to components created with
/// \a hpx#components#distributing_factory.
inline void
initialize_clients(
    hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<random_mem_access::element>& array)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        array.push_back(random_mem_access::element(id));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    {
        // Retrieve the command line options. 
        std::size_t const array_size = vm["array-size"].as<std::size_t>();
        std::size_t const iterations = vm["iterations"].as<std::size_t>();
        std::size_t seed = vm["seed"].as<std::size_t>();

        // If the specified seed is 0, then we pick a random seed.
        if (!seed)
            seed = std::size_t(std::time(0));

        // Seed the C standard libraries random number facilities.
        std::srand(seed);

        hpx::cout << "Seed: " << seed << hpx::endl; 

        ///////////////////////////////////////////////////////////////////////
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        ///////////////////////////////////////////////////////////////////////
        // Array creation.

        // Create a distributing factory. This object can be used to create
        // a distributed array of components. 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the global component type of our element component. 
        hpx::components::component_type type =
            hpx::components::get_component_type<
                random_mem_access::server::element>();

        // Create a global array of element components via distributing factory.
        // The elements will be divided up equally among the localities that
        // supports our element component will 
        std::vector<random_mem_access::element> array;
        initialize_clients(hpx::components::locality_results
            (factory.create_components(type, array_size)), array);

        // Initialize the value of each element with its index in the array. 
        for (std::size_t i = 0; i < array_size; ++i) {
            array[i].init(i);
        }

        ///////////////////////////////////////////////////////////////////////
        // Access phase - increment elements in the array at random.
        std::vector<hpx::lcos::promise<void> > access_phase;

        // Spawn a future for each remote 
        for (std::size_t i = 0; i < iterations; ++i) {
            // Compute the element to access.            
            std::size_t const rn = std::rand() % array_size;
            access_phase.push_back(array[rn].add_async());
        }

        // Wait for the access operations to complete.
        hpx::lcos::wait(access_phase);

        ///////////////////////////////////////////////////////////////////////
        // Print phase - print each element in the array. 
        std::vector<hpx::lcos::promise<void> > print_phase;

        for (std::size_t i = 0; i < array_size; ++i) {
            // Start an asynchronous print action, which will be executed on
            // the locality where each element lives. The I/O, however, will
            // all go to standard output on the console locality.
            print_phase.push_back(array[i].print_async());
        }

        // Wait for all print operations to finish.
        hpx::lcos::wait(print_phase);

        // Print the total walltime that the computation took.
        hpx::cout << "Elapsed time: " << t.elapsed() << " [s]" << hpx::endl;

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
        ("array-size", value<std::size_t>()->default_value(8),
            "the size of the array")
        ("iterations", value<std::size_t>()->default_value(16),
            "the number of lookups to perform")
        ("seed", value<std::size_t>()->default_value(0),
            "the seed for the pseudo random number generator (if 0, a seed "
            "is choosen based on the current system time)")
        ;

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

