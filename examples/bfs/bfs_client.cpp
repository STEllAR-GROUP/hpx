//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "bfs/point.hpp"
#include <hpx/components/distributing_factory/distributing_factory.hpp>

using hpx::util::high_resolution_timer;

inline void
init(hpx::components::server::distributing_factory::iterator_range_type r,
    std::vector<hpx::geometry::point>& accu)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        accu.push_back(hpx::geometry::point(id));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
       high_resolution_timer t;

        const std::size_t num_elements = 5;

        // create a distributing factory locally
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        hpx::components::component_type block_type =
            hpx::components::get_component_type<
                hpx::geometry::point::server_component_type>();

        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, num_elements);

        std::vector<hpx::geometry::point> accu;

        // Initialize the data
        init(locality_results(blocks), accu);


        // Initial Data -----------------------------------------
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<num_elements;i++) {
          // compute the initial velocity so that everything heads to the origin
          initial_phase.push_back(accu[i].init_async(i));
        }

        // vector of gids
        std::vector<hpx::naming::id_type> master_objects;
        for (std::size_t i=0;i<num_elements;i++) {
          master_objects.push_back(accu[i].get_gid());
        }

        // We have to wait for the futures to finish before exiting.
        hpx::lcos::wait(initial_phase);
#if 0
        std::vector<int> neighbors;
        // traverse the graph
        std::size_t root = 0; // define root node
        std::size_t level = 0; // define root node
        std::size_t parent = 9999; // define root node parent
        std::vector<int> neighbors;
        std::vector<hpx::lcos::promise<void> > traverse_phase;
        traverse_phase.push_back(accu[root].traverse_async(level,parent));
        hpx::lcos::wait(traverse_phase,neighbors);
        while (neighbors.size() > 0 ) {
          level++; 
          parent = root;
          BOOST_FOREACH(i,neighbors)
          {
            traverse_phase.push_back(accu[i].traverse_async(level,parent));
          }
          hpx::lcos::wait(traverse_phase,neighbors);
        }
#endif

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;
    } // ensure things are go out of scope

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init("bfs_client", argc, argv); // Initialize and run HPX
}
