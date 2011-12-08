//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "bfs/point.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

/// This function initializes a vector of \a bfs::point clients, 
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<bfs::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(bfs::point(id));
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
        std::size_t const num_elements = vm["n"].as<std::size_t>();
        std::size_t const grainsize = vm["grainsize"].as<std::size_t>();
        std::string const searchfile = vm["searchfile"].as<std::string>();
        std::size_t const max_levels = vm["max-levels"].as<std::size_t>();
        std::size_t const max_num_neighbors
            = vm["max-num-neighbors"].as<std::size_t>();

        std::string const graphfile = vm["graph"].as<std::string>();

        std::size_t ne = num_elements/grainsize;

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type. 
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<bfs::server::point>();

        // Create ne point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, ne);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<bfs::point> points;

        // Populate the client vectors. 
        init(hpx::components::server::locality_results(blocks), points);

        ///////////////////////////////////////////////////////////////////////
        // Read in the graph and the searchfile
        hpx::util::high_resolution_timer readtime;
        std::vector<hpx::lcos::promise<void> > read_phase;

        for (std::size_t i=0;i<ne;i++) {
          read_phase.push_back(points[i].read_async(i,grainsize,max_num_neighbors,graphfile));
        }

        // read in the searchfile containing the root vertices to search
        std::vector<std::size_t> searchroot;
        std::string line;
        std::string val1;
        std::ifstream myfile;
        myfile.open(searchfile);
        if (myfile.is_open()) {
            while (myfile.good()) {
                while (std::getline(myfile,line)) {
                    std::istringstream isstream(line);
                    std::getline(isstream,val1);
                    std::size_t root = boost::lexical_cast<std::size_t>(val1);
                    searchroot.push_back(root);
                }
            }
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation. 
        hpx::lcos::wait(read_phase);
        std::cout << "Elapsed time during read: " << readtime.elapsed() << " [s]" << std::endl;
#if 0
        ///////////////////////////////////////////////////////////////////////
        // KERNEL 1  --- TIMED
        // Initialize the points with the data from the input file. 
        std::vector<hpx::lcos::promise<void> > initial_phase;

        for (std::size_t i=0;i<ne;i++) {
          initial_phase.push_back(points[i].init_async(i,max_num_neighbors,graphfile));
        }

        // While we're waiting for the initialization phase to complete, we 
        // build a vector of all of the point GIDs. This will be used as the
        // input for the next phase.
        std::vector<hpx::naming::id_type> master_objects;
        for (std::size_t i=0;i<ne;i++) {
          master_objects.push_back(points[i].get_gid());
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation. 
        hpx::lcos::wait(initial_phase);

        ///////////////////////////////////////////////////////////////////////
        std::vector<hpx::lcos::promise<std::vector<std::size_t> > > traverse_phase;

        // Traverse the graph.
        std::size_t level = 0; 

        // The root node's parent.
        std::size_t parent = 9999; 

        // Create the parent vectors.
        std::vector<std::vector<std::size_t> > parents;
        for (std::size_t i=0;i<max_levels;i++) {
          parents.push_back(std::vector<std::size_t>());
        }

        std::vector<std::vector<std::size_t> > neighbors,alt_neighbors;

        // Install the root node. 
        parents[level].push_back( root ); 
        traverse_phase.push_back( points[root].traverse_async(level,parent) );

        // Wait for the first part of the traverse phase to complete.
        hpx::lcos::wait(traverse_phase,neighbors);

        for (std::size_t k=1;k<max_levels;k++) {
          // Clear the traversal vector. 
          traverse_phase.resize(0);
  
          if ( (k+1)%2 == 0 ) {
            // Clear the alt_neighbor vector.
            alt_neighbors.resize(0);

            for (std::size_t i=0;i<neighbors.size();i++) {
              // Set the current parent.
              parent = parents[k-1][i];

              for (std::size_t j=0;j<neighbors[i].size();j++) {
                parents[k].push_back( neighbors[i][j] ); 

                // Create a future encapsulating an asynchronous call to
                // the traverse action of bfs::point. 
                traverse_phase.push_back(points[ neighbors[i][j] ].traverse_async(k,parent));
              } 
            }

            // Wait for this phase to finish
            hpx::lcos::wait(traverse_phase,alt_neighbors);

          } else {
            // Clear the neighbor vector.
            neighbors.resize(0);

            for (std::size_t i=0;i<alt_neighbors.size();i++) {
              // Set the current parent.
              parent = parents[k-1][i];

              for (std::size_t j=0;j<alt_neighbors[i].size();j++) {
                parents[k].push_back( alt_neighbors[i][j] ); 

                // Create a future encapsulating an asynchronous call to
                // the traverse action of bfs::point. 
                traverse_phase.push_back(points[ alt_neighbors[i][j] ].traverse_async(k,parent));
              } 
            }

            // Wait for this phase to finish
            hpx::lcos::wait(traverse_phase,neighbors);
          }
        }
#endif
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
        ("n", value<std::size_t>()->default_value(20000),
            "the number of nodes in the graph")
        ("grainsize", value<std::size_t>()->default_value(500),
            "the grainsize of the components")
        ("max-num-neighbors", value<std::size_t>()->default_value(20),
            "the maximum number of neighbors")
        ("max-levels", value<std::size_t>()->default_value(20),
            "the maximum number of levels to traverse")
        ("searchfile", value<std::string>()->default_value("g10_search.txt"),
            "the file containing the roots to search in the graph")
        ("graph", value<std::string>()->default_value("g10.txt"),
            "the file containing the graph");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

