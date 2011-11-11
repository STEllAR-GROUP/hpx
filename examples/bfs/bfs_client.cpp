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

       const std::size_t maxlevels = 20;
       std::size_t num_elements = 5;
       if (vm.count("n"))
        num_elements = vm["n"].as<std::size_t>();

       // define the root node
       std::size_t root = 0;
       if (vm.count("root"))
        root = vm["root"].as<std::size_t>();
  
       std::string graphfile;
       if (vm.count("graph"))
        graphfile = vm["graph"].as<std::string>();

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
          initial_phase.push_back(accu[i].init_async(i,graphfile));
        }

        // vector of gids
        std::vector<hpx::naming::id_type> master_objects;
        for (std::size_t i=0;i<num_elements;i++) {
          master_objects.push_back(accu[i].get_gid());
        }

        // We have to wait for the futures to finish before exiting.
        hpx::lcos::wait(initial_phase);

        // traverse the graph
        std::size_t level = 0; 
        std::size_t parent = 9999; // define root node parent
        std::vector< std::vector<std::size_t> >  parents;
        for (std::size_t i=0;i<maxlevels;i++) {
          parents.push_back(std::vector<std::size_t>() );
        }
        std::vector< std::vector<std::size_t> > neighbors,alt_neighbors;
        std::vector<hpx::lcos::promise<std::vector<std::size_t> > > traverse_phase;

        parents[level].push_back( root ); 
        traverse_phase.push_back(accu[root].traverse_async(level,parent));
        hpx::lcos::wait(traverse_phase,neighbors);

        // the rest
        for (std::size_t k=1;k<maxlevels;k++) {
          traverse_phase.resize(0);
  
          if ( (k+1)%2 == 0 ) {
            alt_neighbors.resize(0);
            for (std::size_t i=0;i<neighbors.size();i++) {
              parent = parents[k-1][i];
              for (std::size_t j=0;j<neighbors[i].size();j++) {
                parents[k].push_back( neighbors[i][j] ); 
                traverse_phase.push_back(accu[ neighbors[i][j] ].traverse_async(k,parent));
              } 
            }
            hpx::lcos::wait(traverse_phase,alt_neighbors);
          } else {
            neighbors.resize(0);
            for (std::size_t i=0;i<alt_neighbors.size();i++) {
              parent = parents[k-1][i];
              for (std::size_t j=0;j<alt_neighbors[i].size();j++) {
                parents[k].push_back( alt_neighbors[i][j] ); 
                traverse_phase.push_back(accu[ alt_neighbors[i][j] ].traverse_async(k,parent));
              } 
            }
            hpx::lcos::wait(traverse_phase,neighbors);
          }
        }

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
        ("n", value<std::size_t>()->default_value(100),
            "the number of nodes in the graph")
        ("root", value<std::size_t>()->default_value(0),
            "define the root node in the graph")
        ("graph", value<std::string>()->default_value("g1.txt"),
            "the file containing the graph");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX

}
