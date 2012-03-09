//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>

#include "bfs_graph/graph.hpp"
#include "bfs_graph/bgl_graph.hpp"
#include "bfs_graph/concurrent_bgl_graph.hpp"

#include "bfs_graph_validate.hpp"
#include "bfs_graph_util.hpp"
#include "bfs_graph_single_locality.hpp"

namespace bfs_graph { namespace single_locality
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename GraphComponent>
    double kernel1(std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::vector<GraphComponent>& graphs)
    {
        hpx::util::high_resolution_timer kernel1time;

        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our graph component.
        hpx::components::component_type block_type =
            GraphComponent::get_component_type();

        // get a unique list of nodes - this is what determines num_graphs
        // find the largest node number
        std::size_t num_elements = std::accumulate(
            edgelist.begin(), edgelist.end(), std::size_t(0), max_node) + 1;

        // determine the number of elements per graph instance
        std::size_t num_graphs = 1;
    //     if (0 != grainsize)
    //     {
    //         if (num_elements % grainsize)
    //             num_elements += grainsize - (num_elements % grainsize);
    //         num_graphs = num_elements/grainsize;
    //     }

        // Create 'num_elements' graph components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, num_graphs);

        // This vector will hold client classes referring to all of the
        // components we just created.
        BOOST_FOREACH(hpx::naming::id_type const& id,
            hpx::components::server::locality_results(blocks))
        {
            graphs.push_back(GraphComponent(id));
        }

        // Put the graph in the data structure
        std::vector<hpx::lcos::future<void> > init_phase;
        for (std::size_t i = 0; i < num_graphs; ++i)
            init_phase.push_back(graphs[i].init_async(i, num_elements, edgelist));
        hpx::lcos::wait(init_phase);

        return kernel1time.elapsed();
    }

    ///////////////////////////////////////////////////////////////////////////////
    template <typename GraphComponent>
    void bfs(std::string const& desc, bool validate, std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::vector<std::size_t> const& searchroots)
    {
        // Print time statistics
        std::string s = " Statistics for " + desc;
        std::string delim(s.size()+1, '-');
        std::cout << std::endl << delim
                  << std::endl << s
                  << std::endl << delim
                  << std::endl;

        // KERNEL 1 --- TIMED
        std::vector<GraphComponent> graphs;
        double kernel1_time = kernel1(grainsize, edgelist, graphs);
        std::cout << " kernel 1:             " << kernel1_time
                  << std::endl << std::endl;

        // KERNEL 2 --- TIMED
        std::cout << " kernel 2:             "  << std::endl;
        std::vector<double> kernel2_time(searchroots.size(), 0.0);
        std::vector<std::size_t> kernel2_nedge(searchroots.size(), std::size_t(0));

        // go through each root position
        bool validation = true;
        for (std::size_t step = 0; step < searchroots.size(); ++step) {
            // do BFS, bfs() returns timing
            kernel2_time[step] = graphs[0].bfs(searchroots[step]);

            // Validate  -- Not timed
            if (validate) {
                std::vector<std::size_t> parents(graphs[0].get_parents());

                std::size_t num_nodes = 0;
                int rc = bfs_graph::validate_graph(searchroots[step], parents,
                    edgelist, num_nodes);
                if (rc < 0) {
                    validation = false;
                    std::cout << " Validation failed for searchroot: "
                              << searchroots[step] << " (rc: " << rc << ")"
                              << std::endl;
                    break;
                }
                kernel2_nedge[step] = num_nodes;
            }

            // Reset for the next root
            std::vector<hpx::lcos::future<void> > reset_phase;
            for (std::size_t i = 0; i < graphs.size(); ++i)
                reset_phase.push_back(graphs[i].reset_async());
            hpx::lcos::wait(reset_phase);
        }

        if (validation && validate)
            bfs_graph::print_statistics(kernel2_time, kernel2_nedge);
    }

    ///////////////////////////////////////////////////////////////////////////
    // start benchmarking: single threaded (vanilla) BFS
    void run_benchmarks(bool validate, std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::vector<std::size_t> const& searchroots)
    {
        // single threaded (vanilla) BFS
        bfs<bfs::graph>("Single threaded BFS",
            validate, grainsize, edgelist, searchroots);

//         // single threaded (BGL) BFS
//         bfs<bfs::bgl_graph>("Single threaded BFS (based on BGL)",
//             validate, grainsize, edgelist, searchroots);

        // multi threaded (BGL) BFS
        bfs<bfs::concurrent_bgl_graph>("Multi threaded BFS (based on BGL)",
            validate, grainsize, edgelist, searchroots);
    }
}}
