//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "graph500/point.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

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

        // Generate the search roots
        //  This part isn't functional yet (need C support); 
        //   the following is a stop-gap measure
        std::string const searchfile = "g10_search.txt";
        std::vector<std::size_t> searchroot;
        {
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
                  // increment all nodes and neighbors by 1; the smallest edge number is 1
                  // edge 0 is reserved for the parent of the root and for unvisited edges
                  searchroot.push_back(root+1);
              }
            }
          }
        }

        ///////////////////////////////////////////////////////////////////////
        // Put the graph in the data structure
        std::vector<hpx::lcos::promise<void> > init_phase;

        for (std::size_t i=0;i<number_partitions;i++) {
          init_phase.push_back(points[i].init_async(i,scale,number_partitions));
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation.
        hpx::lcos::wait(init_phase);
        double kernel1_time = kernel1time.elapsed();
        std::cout << "Elapsed time during kernel 1: " << kernel1_time << " [s]" << std::endl;

        // Begin Kernel 2
        std::vector<double> kernel2_time;
        std::vector<double> kernel2_nedge;
        kernel2_time.resize(searchroot.size());
        kernel2_nedge.resize(searchroot.size());

        hpx::util::high_resolution_timer part_kernel2time;
        std::vector<hpx::lcos::promise<void> > bfs_phase;

        for (std::size_t i=0;i<number_partitions;i++) {
          bfs_phase.push_back(points[i].bfs_async());
        }
        hpx::lcos::wait(bfs_phase);
        double part_kernel2_time = part_kernel2time.elapsed();
        part_kernel2_time /= searchroot.size();

        std::vector<std::size_t> startup_neighbor;
        startup_neighbor.resize(1);
        for (std::size_t step=0;step<searchroot.size();step++) {
          hpx::util::high_resolution_timer kernel2time;
          {  // tighten up the edges for each root
            std::vector<hpx::lcos::promise<void> > merge_phase;
            startup_neighbor[0] = searchroot[step];
            for (std::size_t i=0;i<number_partitions;i++) {
              merge_phase.push_back(points[i].merge_graph_async(searchroot[step],startup_neighbor));
            }
            hpx::lcos::wait(merge_phase);
          }
          kernel2_time[step] = kernel2time.elapsed() + part_kernel2_time;

          // validate

          // reset tightening
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

