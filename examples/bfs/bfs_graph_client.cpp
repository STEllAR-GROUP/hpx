//  Copyright (c) 2007-2011 Matthew Anderson
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <boost/foreach.hpp>

#include <vector>
#include <set>

#include "bfs_graph/graph.hpp"

///////////////////////////////////////////////////////////////////////////////
// this routine mirrors the matlab validation routine
int validate(std::size_t searchkey,
    std::vector<std::size_t> const& parents,
    std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
    std::size_t& num_nodes);

///////////////////////////////////////////////////////////////////////////////
void get_statistics(std::vector<double> x, double &minimum, double &mean,
    double &stdev, double &firstquartile,
    double &median, double &thirdquartile, double &maximum);

///////////////////////////////////////////////////////////////////////////////
bool read_edge_list(std::string const& graphfile,
    std::vector<std::pair<std::size_t, std::size_t> >& edgelist)
{
    std::ifstream myfile(graphfile.c_str());
    if (myfile.is_open()) {
        std::size_t node, neighbor;
        while (myfile >> node >> neighbor)
            edgelist.push_back(std::make_pair(node+1, neighbor+1));
        return myfile.eof();
    }

    std::cerr << " File " << graphfile
              << " not found! Exiting... " << std::endl;
    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool read_search_node_list(std::string const& searchfile,
    std::vector<std::size_t>& searchroots)
{
    std::ifstream myfile(searchfile.c_str());
    if (myfile.is_open()) {
        std::size_t root;
        while (myfile >> root)
            searchroots.push_back(root+1);
        return myfile.eof();
    }

    std::cerr << " File " << searchfile
              << " not found! Exiting... " << std::endl;
    return false;
}

///////////////////////////////////////////////////////////////////////////////
inline std::size_t
max_node(std::size_t n1, std::pair<std::size_t, std::size_t> const& p)
{
    return (std::max)((std::max)(n1, p.first), p.second);
}

double kernel1(std::size_t grainsize,
    std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
    std::vector<bfs::graph>& graphs)
{
    hpx::util::high_resolution_timer kernel1time;

    // Create a distributing factory locally. The distributing factory can
    // be used to create blocks of components that are distributed across
    // all localities that support that component type.
    hpx::components::distributing_factory factory;
    factory.create(hpx::find_here());

    // Get the component type for our graph component.
    hpx::components::component_type block_type =
        hpx::components::get_component_type<bfs::server::graph>();

    // get a unique list of nodes - this is what determines num_graphs
    // find the largest node number
    std::size_t num_elements = std::accumulate(edgelist.begin(), edgelist.end(),
        std::size_t(0), max_node) + 1;

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
        graphs.push_back(bfs::graph(id));
    }

    // Put the graph in the data structure
    std::vector<hpx::lcos::promise<void> > init_phase;
    for (std::size_t i = 0; i < num_graphs; ++i)
        init_phase.push_back(graphs[i].init_async(i, num_elements, edgelist));
    hpx::lcos::wait(init_phase);

    return kernel1time.elapsed();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map &vm)
{
    {
        // Start a high resolution timer to record the execution time of this
        // example.
        hpx::util::high_resolution_timer t;

        // Retrieve the command line options.
        std::size_t grainsize = 0;
        if (vm.count("grainsize"))
            grainsize = vm["grainsize"].as<std::size_t>();

//         std::size_t const max_levels = vm["max-levels"].as<std::size_t>();
//         std::size_t const max_num_neighbors
//             = vm["max-num-neighbors"].as<std::size_t>();

        bool const validator = vm["validator"].as<bool>();
        std::string const graphfile = vm["graph"].as<std::string>();
        std::string const searchfile = vm["searchfile"].as<std::string>();

        // read in the searchfile containing the edge definitions
        std::vector<std::pair<std::size_t, std::size_t> > edgelist;
        if (!read_edge_list(graphfile, edgelist))
        {
            hpx::finalize();
            return -1;
        }

        // read in the searchfile containing the root vertices to search
        std::vector<std::size_t> searchroots;
        if (!read_search_node_list(searchfile, searchroots))
        {
            hpx::finalize();
            return -1;
        }
        std::cout << "Elapsed time during read: " << t.elapsed()
                  << " [s]" << std::endl;

        // KERNEL 1 --- TIMED
        std::vector<bfs::graph> graphs;
        double kernel1_time = kernel1(grainsize, edgelist, graphs);
        std::cout << "Elapsed time during kernel 1: " << kernel1_time
                  << " [s]" << std::endl;

        // KERNEL 2 --- TIMED
        std::vector<double> kernel2_time(searchroots.size(), 0.0);
        std::vector<double> kernel2_nedge(searchroots.size(), 0.0);

        // go through each root position
        bool validation = true;
        for (std::size_t step = 0; step < searchroots.size(); ++step) {
            // do BFS, bfs() returns timing
            kernel2_time[step] = graphs[0].bfs(searchroots[step]);

            // Validate  -- Not timed
            if (validator) {
                std::vector<std::size_t> parents(graphs[0].get_parents());

                std::size_t num_nodes = 0;
                int rc = validate(searchroots[step], parents, edgelist, num_nodes);
                if (rc < 0) {
                    validation = false;
                    std::cout << " Validation failed for searchroot: "
                              << searchroots[step] << " (rc: " << rc << ")"
                              << std::endl;
                    break;
                }
                kernel2_nedge[step] = static_cast<double>(num_nodes);
            }

            // Reset for the next root
            std::vector<hpx::lcos::promise<void> > reset_phase;
            for (std::size_t i = 0; i < graphs.size(); ++i)
                reset_phase.push_back(graphs[i].reset_async());
            hpx::lcos::wait(reset_phase);
        }

        if (validation && validator) {
            // Prep output statistics
            double minimum, mean, stdev;
            double firstquartile, median, thirdquartile, maximum;
            get_statistics(kernel2_time, minimum, mean, stdev,
                firstquartile, median, thirdquartile, maximum);

            // Print time statistics
            std::cout << " construction_time:    " << kernel1_time << std::endl;

            std::cout << " min_time:             " << minimum << std::endl;
            std::cout << " firstquartile_time:   " << firstquartile << std::endl;
            std::cout << " median_time:          " << median << std::endl;
            std::cout << " thirdquartile_time:   " << thirdquartile << std::endl;
            std::cout << " max_time:             " << maximum << std::endl;
            std::cout << " mean_time:            " << mean << std::endl;
            std::cout << " stddev_time:          " << stdev << std::endl;

            double n_min, n_mean, n_stdev;
            double n_firstquartile, n_median, n_thirdquartile, n_maximum;
            get_statistics(kernel2_nedge, n_min, n_mean, n_stdev,
                n_firstquartile, n_median, n_thirdquartile, n_maximum);

            std::cout << " min_nedge:            " << n_min << std::endl;
            std::cout << " firstquartile_nedge:  " << n_firstquartile << std::endl;
            std::cout << " median_nedge:         " << n_median << std::endl;
            std::cout << " thirdquartile_nedge:  " << n_thirdquartile << std::endl;
            std::cout << " max_nedge:            " << n_maximum << std::endl;
            std::cout << " mean_nedge:           " << n_mean << std::endl;
            std::cout << " stddev_nedge:         " << n_stdev << std::endl;

            std::vector<double> TEPS;
            TEPS.resize(kernel2_nedge.size());
            for (std::size_t i = 0; i < kernel2_nedge.size(); ++i)
                TEPS[i] = kernel2_nedge[i]/kernel2_time[i];

            std::size_t N = TEPS.size();
            double TEPS_min, TEPS_mean, TEPS_stdev;
            double TEPS_firstquartile, TEPS_median, TEPS_thirdquartile, TEPS_maximum;
            get_statistics(TEPS, TEPS_min, TEPS_mean, TEPS_stdev,
                TEPS_firstquartile, TEPS_median, TEPS_thirdquartile, TEPS_maximum);

            // Harmonic standard deviation from:
            // Nilan Norris, The Standard Errors of the Geometric and Harmonic
            // Means and Their Application to Index Numbers, 1940.
            // http://www.jstor.org/stable/2235723
            TEPS_stdev = TEPS_stdev/(TEPS_mean*TEPS_mean*sqrt( (double) (N-1) ) );

            std::cout << " min_TEPS:             " << TEPS_min << std::endl;
            std::cout << " firstquartile_TEPS:   " << TEPS_firstquartile << std::endl;
            std::cout << " median_TEPS:          " << TEPS_median << std::endl;
            std::cout << " thirdquartile_TEPS:   " << TEPS_thirdquartile << std::endl;
            std::cout << " max_TEPS:             " << TEPS_maximum << std::endl;
            std::cout << " harmonic_mean_TEPS:   " << TEPS_mean << std::endl;
            std::cout << " harmonic_stddev_TEPS: " << TEPS_stdev << std::endl;
        }

        // Print the total wall-time that the computation took.
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
        ("grainsize", "the grainsize of the components")
        ("max-num-neighbors", value<std::size_t>()->default_value(20),
            "the maximum number of neighbors")
        ("max-levels", value<std::size_t>()->default_value(10),
            "the maximum number of levels to traverse")
        ("searchfile", value<std::string>()->default_value("g10_search.txt"),
            "the file containing the roots to search in the graph")
        ("graph", value<std::string>()->default_value("g10.txt"),
            "the file containing the graph")
        ("validator", value<bool>()->default_value(true),
            "whether to run the validation (slow)");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

