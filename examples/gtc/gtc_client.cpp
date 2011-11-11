//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "point/point.hpp"
#include "particle/particle.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

using hpx::util::high_resolution_timer;

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::point>& accu_points)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu_points.push_back(hpx::geometry::point(id));
    }
}

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::particle>& accu_particle)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu_particle.push_back(hpx::geometry::particle(id));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
       high_resolution_timer t;

       std::size_t num_gridpoints = 5;
       std::size_t num_particles = 5;
       if (vm.count("n"))
        num_gridpoints = vm["n"].as<std::size_t>();

       if (vm.count("np"))
        num_particles = vm["np"].as<std::size_t>();

       std::string meshfile;
       if (vm.count("mesh"))
        meshfile = vm["mesh"].as<std::string>();

       std::string particlefile;
       if (vm.count("particles"))
        particlefile = vm["particles"].as<std::string>();

        // ---------------------------------------------------------
        // create a distributing factory locally for the gridpoints 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        hpx::components::component_type block_type =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        hpx::components::distributing_factory::result_type blocks_points =
            factory.create_components(block_type, num_gridpoints);

        // create a distributing factory locally for the particles 
        hpx::components::distributing_factory factory_particles;
        factory_particles.create(hpx::find_here());

        hpx::components::component_type block_type_particles =
            hpx::components::get_component_type<
                hpx::geometry::particle::server_component_type>();

        hpx::components::distributing_factory::result_type blocks_particles =
            factory_particles.create_components(block_type_particles, num_particles);
        // ---------------------------------------------------------

        std::vector<hpx::geometry::point> accu_points;
        std::vector<hpx::geometry::particle> accu_particles;

        // Initialize the data -- both mesh and particles
        init(locality_results(blocks_points), accu_points);
        init(locality_results(blocks_particles), accu_particles);


        // Initial Data -----------------------------------------
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<num_gridpoints;i++) {
          initial_phase.push_back(accu_points[i].init_async(i,meshfile));
        }

        for (std::size_t i=0;i<num_particles;i++) {
          initial_phase.push_back(accu_particles[i].init_async(i,particlefile));
        }

        // vector gids of the particle components
        std::vector<hpx::naming::id_type> particle_components;
        for (std::size_t i=0;i<num_particles;i++) {
          particle_components.push_back(accu_particles[i].get_gid());
        }

        // We have to wait for the futures to finish before exiting.
        hpx::lcos::wait(initial_phase);

        // Start the search/charge depositing phase
        std::vector<hpx::lcos::promise<int> > charge_phase;

        for (std::size_t i=0;i<num_gridpoints;i++) {
          charge_phase.push_back(accu_points[i].search_async(particle_components));
        }
        std::vector<int> search_vector;
        hpx::lcos::wait(charge_phase,search_vector);

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;
    } // ensure things are go out of scope

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
