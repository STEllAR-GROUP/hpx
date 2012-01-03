//  Copyright (c) 2007-2011 Matthew Anderson
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <vector>

#include "bfs_graph_util.hpp"
#include "bfs_graph_single_locality.hpp"

///////////////////////////////////////////////////////////////////////////////
int do_work(boost::program_options::variables_map &vm)
{
    // Retrieve the command line options.
    std::size_t grainsize = 0;
    if (vm.count("grainsize"))
        grainsize = vm["grainsize"].as<std::size_t>();

//         std::size_t const max_levels = vm["max-levels"].as<std::size_t>();
//         std::size_t const max_num_neighbors
//             = vm["max-num-neighbors"].as<std::size_t>();

    bool const validate = vm["validate"].as<bool>();
    std::string const graphfile = vm["graph"].as<std::string>();
    std::string const searchfile = vm["searchfile"].as<std::string>();

    // Start a high resolution timer to record the execution time of this
    // example.
    hpx::util::high_resolution_timer t;

    // read in the searchfile containing the edge definitions
    std::vector<std::pair<std::size_t, std::size_t> > edgelist;
    if (!bfs_graph::read_edge_list(graphfile, edgelist))
        return -1;

    // read in the searchfile containing the root vertices to search
    std::vector<std::size_t> searchroots;
    if (!bfs_graph::read_search_node_list(searchfile, searchroots))
        return -1;

    std::cout << "Elapsed time during read: " << t.elapsed()
              << " [s]" << std::endl;

    // start benchmarking: single threaded BFS variants
    bfs_graph::single_locality::run_benchmarks(
        validate, grainsize, edgelist, searchroots);

    // Print the total wall-time that the computation took.
    std::cout  << std::endl << "Elapsed time: " << t.elapsed()
               << " [s]" << std::endl;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    int result = do_work(vm);
    hpx::finalize();
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::value;

    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("grainsize", "the grainsize of the components")
        ("max-num-neighbors", value<std::size_t>()->default_value(20),
            "the maximum number of neighbors")
        ("max-levels", value<std::size_t>()->default_value(10),
            "the maximum number of levels to traverse")
        ("searchfile", value<std::string>()->default_value("g10_search.txt"),
            "the file containing the roots to search in the graph")
        ("graph", value<std::string>()->default_value("g10.txt"),
            "the file containing the graph")
        ("validate", value<bool>()->default_value(true),
            "whether to run the validation (slow)");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

