//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point/point.hpp"
#include "particle/particle.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

/// This function initializes a vector of \a hpx#geometry#point clients, 
/// connecting them to components created with
/// \a hpx#components#distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::point>& point)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        point.push_back(hpx::geometry::point(id));
    }
}

/// This function initializes a vector of \a hpx#geometry#particle clients, 
/// connecting them to components created with
/// \a hpx#components#distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::particle>& particle)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        particle.push_back(hpx::geometry::particle(id));
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
        std::size_t num_gridpoints = vm["n"].as<std::size_t>();
        std::size_t num_particles = vm["np"].as<std::size_t>();
        std::string meshfile = vm["mesh"].as<std::string>();
        std::string particlefile = vm["particles"].as<std::string>();

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally for the gridpoints. The
        // distributing factory can be used to create a block of components
        // that are distributed across all localities that support that
        // component type. 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type_points =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        // Create num_gridpoints point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_points =
            factory.create_components(block_type_points, num_gridpoints);

        // Get the component type for our particle component.
        hpx::components::component_type block_type_particles =
            hpx::components::get_component_type<
                hpx::geometry::particle::server_component_type>();

        // Create num_gridpoints particle components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks_particles =
            factory.create_components(block_type_particles, num_particles);

        ///////////////////////////////////////////////////////////////////////
        // These two vectors will hold client classes referring to all of the
        // components we just created.
        std::vector<hpx::geometry::point> points;
        std::vector<hpx::geometry::particle> particles;

        // Populate the client vectors. 
        init(locality_results(blocks_points), points);
        init(locality_results(blocks_particles), particles);

        ///////////////////////////////////////////////////////////////////////
        // Initialize the particles and points with the data from the input
        // files. 
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<num_gridpoints;i++) {
          initial_phase.push_back(points[i].init_async(i,meshfile));
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
        std::vector<hpx::lcos::promise<int> > charge_phase;

        // We use the vector of particle component GIDS that we created during
        // the initialization phase as the input to the search action on each
        // point. 
        for (std::size_t i=0;i<num_gridpoints;i++) {
          charge_phase.push_back(points[i].search_async(particle_components));
        }

        // Our results will get copied into this vector.
        std::vector<int> search_vector;

        // Wait for the search/charge depositing phase to complete.
        hpx::lcos::wait(charge_phase,search_vector);

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

    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("n", value<std::size_t>()->default_value(5),
            "the number of gridpoints in the mesh")
        ("np", value<std::size_t>()->default_value(5),
            "the number of particles in the mesh")
        ("mesh", value<std::string>()->default_value("mesh.txt"),
            "the file containing the mesh")
        ("particles", value<std::string>()->default_value("particles.txt"),
            "the file containing the particles");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX
}

