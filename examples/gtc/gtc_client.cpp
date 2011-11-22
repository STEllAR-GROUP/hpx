//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point/point.hpp"
#include "particle/particle.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

/// This function initializes a vector of \a gtc::point clients, 
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<gtc::point>& point)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        point.push_back(gtc::point(id));
    }
}

/// This function initializes a vector of \a gtc::particle clients, 
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<gtc::particle>& particle)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        particle.push_back(gtc::particle(id));
    }
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
        std::size_t const num_gridpoints = vm["n"].as<std::size_t>();
        std::size_t const num_particles = vm["np"].as<std::size_t>();
        std::size_t const max_num_neighbors
            = vm["max-num-neighbors"].as<std::size_t>();

        std::string const meshfile = vm["mesh"].as<std::string>();
        std::string const particlefile = vm["particles"].as<std::string>();

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type. 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the global component type of our point component.
        hpx::components::component_type block_type_points =
            hpx::components::get_component_type<gtc::server::point>();

        // Create num_gridpoints point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_points =
            factory.create_components(block_type_points, num_gridpoints);

        // Get the component type for our particle component.
        hpx::components::component_type block_type_particles =
            hpx::components::get_component_type<gtc::server::particle>();

        // Create num_gridpoints particle components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_particles =
            factory.create_components(block_type_particles, num_particles);

        ///////////////////////////////////////////////////////////////////////
        // These two vectors will hold client classes referring to all of the
        // components we just created.
        std::vector<gtc::point> points;
        std::vector<gtc::particle> particles;

        // Populate the client vectors. 
        init(hpx::components::server::locality_results(blocks_points), points);
        init(hpx::components::server::locality_results(blocks_particles), particles);

        ///////////////////////////////////////////////////////////////////////
        // Initialize the particles and points with the data from the input
        // files. 
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<num_gridpoints;i++) {
          initial_phase.push_back(points[i].init_async(i,max_num_neighbors,meshfile));
        }

        for (std::size_t i=0;i<num_particles;i++) {
          initial_phase.push_back(particles[i].init_async(i,particlefile));
        }

        // While we're waiting for the initialization phase to complete, we 
        // build a vector of all of the particle GIDs. This will be used as the
        // input for the next phase.
        std::vector<hpx::naming::id_type> particle_components;
        for (std::size_t i=0;i<num_particles;i++) {
          particle_components.push_back(particles[i].get_gid());
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation. 
        hpx::lcos::wait(initial_phase);

        ///////////////////////////////////////////////////////////////////////
        // Start the search/charge depositing phase.
        std::vector<hpx::lcos::promise<void> > charge_phase;

        // We use the vector of particle component GIDS that we created during
        // the initialization phase as the input to the search action on each
        // point. 
        for (std::size_t i=0;i<num_gridpoints;i++) {
          charge_phase.push_back(points[i].search_async(particle_components));
        }

        // Wait for the search/charge depositing phase to complete.
        hpx::lcos::wait(charge_phase);

        // Print the total walltime that the computation took.
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
        ("n", value<std::size_t>()->default_value(5),
            "the number of gridpoints")
        ("np", value<std::size_t>()->default_value(5),
            "the number of particles")
        ("max-num-neighbors", value<std::size_t>()->default_value(20),
            "the maximum number of neighbors")
        ("mesh", value<std::string>()->default_value("mesh.txt"),
            "the file containing the mesh")
        ("particles", value<std::string>()->default_value("particles.txt"),
            "the file containing the particles");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

