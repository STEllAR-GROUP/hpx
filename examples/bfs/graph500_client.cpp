//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "graph500/point.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

void make_random_numbers(
       /* in */ int64_t nvalues    /* Number of values to generate */,
       /* in */ uint64_t userseed1 /* Arbitrary 64-bit seed value */,
       /* in */ uint64_t userseed2 /* Arbitrary 64-bit seed value */,
       /* in */ int64_t position   /* Start index in random number stream */,
       /* out */ double* result    /* Returned array of values */
);

/// This function initializes a vector of \a graph500::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<graph500::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(graph500::point(id));
    }
}

// this routine mirrors the matlab validation routine
int validate(std::vector<std::size_t> const& parent,
             std::vector<std::size_t> const& levels,
             std::vector<std::size_t> const& parentindex,
             std::vector<std::size_t> const& nodelist,
             std::vector<std::size_t> const& neighborlist,
             std::size_t searchkey,std::size_t &num_edges);

void get_statistics(std::vector<double> const& x, double &minimum, double &mean,
                    double &stdev, double &firstquartile,
                    double &median, double &thirdquartile, double &maximum);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        ///////////////////////////////////////////////////////////////////////
        // Retrieve the command line options.
        std::size_t const number_partitions = vm["number_partitions"].as<std::size_t>();
        std::size_t const scale = vm["scale"].as<std::size_t>();
        bool const validator = vm["validator"].as<bool>();

        ///////////////////////////////////////////////////////////////////////
        // KERNEL 1  --- TIMED 
        hpx::util::high_resolution_timer kernel1time;

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<graph500::server::point>();

        // ---------------------------------------------------------------
        // Create ne point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, number_partitions);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<graph500::point> points;

        // Populate the client vectors.
        init(hpx::components::server::locality_results(blocks), points);
        
        ///////////////////////////////////////////////////////////////////////
        // Get the gids of each component
        std::vector<hpx::naming::id_type> point_components;
        for (std::size_t i=0;i<number_partitions;i++) {
          point_components.push_back(points[i].get_gid());
        }

        ///////////////////////////////////////////////////////////////////////
        // Put the graph in the data structure
        std::vector<hpx::lcos::promise<void> > init_phase;

        for (std::size_t i=0;i<number_partitions;i++) {
          init_phase.push_back(points[i].init_async(i,scale,number_partitions,point_components));
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation.
        hpx::lcos::wait(init_phase);
        double kernel1_time = kernel1time.elapsed();
        std::cout << "Elapsed time during kernel 1: " << kernel1_time << " [s]" << std::endl;
        // Generate the search roots
        std::vector<std::size_t> bfs_roots;
        bfs_roots.resize(64);  // the graph500 specifies 64 search roots 
                                // must be used

        {  // generate the bfs roots
          int64_t nglobalverts = (int64_t)(1) << scale;
          uint64_t counter = 0;
          uint64_t seed1 = 2;
          uint64_t seed2 = 3;
          for (std::size_t bfs_root_idx=0;bfs_root_idx<bfs_roots.size();bfs_root_idx++) {
            int64_t root;
            while (1) {
              double d[2];
              make_random_numbers(2, seed1, seed2, counter, d);
              root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
              counter += 2;
              if ( counter > (uint64_t) 2 * nglobalverts) break;
              int is_duplicate = 0;
              for (std::size_t i = 0; i < bfs_root_idx; ++i) {
                if ( (std::size_t) root == bfs_roots[i]) {
                  is_duplicate = 1;
                  break;
                }
              }
              if (is_duplicate) continue; /* Everyone takes the same path here */
              int root_ok = 0;
              // check if the root is in the graph; if so, set root_ok to be true
              {
                std::size_t test_root = (std::size_t) root;
                std::vector<bool> search_vector;
                std::vector<hpx::lcos::promise<bool> > has_edge_phase;
                for (std::size_t i=0;i<number_partitions;i++) {
                  has_edge_phase.push_back(points[i].has_edge_async(test_root));
                }
                hpx::lcos::wait(has_edge_phase,search_vector);
                for (std::size_t jj=0;jj<search_vector.size();jj++) {
                  if ( search_vector[jj] ) {
                    root_ok = 1;
                    break;
                  }
                }
              }
              if (root_ok) break;
  
            }          
            bfs_roots[bfs_root_idx] = root;
          }
        }

        // Begin Kernel 2
        std::vector<double> kernel2_time;
        std::vector<double> kernel2_nedge;
        kernel2_time.resize(bfs_roots.size());
        kernel2_nedge.resize(bfs_roots.size());

        hpx::util::high_resolution_timer part_kernel2time;
        std::vector<hpx::lcos::promise<void> > bfs_phase;

        for (std::size_t i=0;i<number_partitions;i++) {
          bfs_phase.push_back(points[i].bfs_async());
        }
        hpx::lcos::wait(bfs_phase);
        double part_kernel2_time = part_kernel2time.elapsed();
        part_kernel2_time /= bfs_roots.size();

        for (std::size_t step=0;step<bfs_roots.size();step++) {
          hpx::util::high_resolution_timer kernel2time;

          std::vector<std::vector<vertex_data> > merge_result;
          {  // tighten up the edges for each root
            std::vector<hpx::lcos::promise<std::vector<vertex_data> > > merge_phase;
            std::vector<vertex_data> startup;
            std::vector<std::size_t> startup_neighbor;
            startup_neighbor.push_back(bfs_roots[step]);
            vertex_data data;
            data.node = bfs_roots[step];
            data.neighbors = startup_neighbor;
            startup.push_back(data);
            for (std::size_t i=0;i<number_partitions;i++) {
              merge_phase.push_back(points[i].merge_graph_async(startup));
            }
            hpx::lcos::wait(merge_phase,merge_result);
          }

          // next levels
          for (std::size_t level=1;level<3;level++) {  
            // tighten up the edges for each root
            std::vector<hpx::lcos::promise<std::vector<vertex_data> > > merge_phase;
            for (std::size_t j=0;j<merge_result.size();j++) {
              for (std::size_t i=0;i<number_partitions;i++) {
                if ( i != j ) {
                  merge_phase.push_back(points[i].merge_graph_async(merge_result[j]));
                }
              }
            }
            merge_result.resize(0);
            hpx::lcos::wait(merge_phase,merge_result);
          }

          kernel2_time[step] = kernel2time.elapsed() + part_kernel2_time;

          // validate

          {  // reset or loosening phase (make individual graphs disjoint again)
            std::vector<hpx::lcos::promise<void> > reset_phase;
            for (std::size_t i=0;i<number_partitions;i++) {
              reset_phase.push_back(points[i].reset_async());
            }
            hpx::lcos::wait(reset_phase);
          }
        } 

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
        ("number_partitions", value<std::size_t>()->default_value(10),
            "the number of components")
        ("scale", value<std::size_t>()->default_value(10),
            "the scale of the graph problem size assuming edge factor 16")
        ("validator", value<bool>()->default_value(true),
            "whether to run the validation (slow)");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

