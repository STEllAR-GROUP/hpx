//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "bfs_partitioned/point.hpp"
#include <boost/numeric/ublas/vector_sparse.hpp>

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

bool fexists(std::string const filename)
{
  std::ifstream ifile(filename);
  return ifile ? true : false;
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
        std::size_t const grainsize = vm["grainsize"].as<std::size_t>();
        std::string const searchfile = vm["searchfile"].as<std::string>();
        std::size_t const max_levels = vm["max-levels"].as<std::size_t>();
        bool const validator = vm["validator"].as<bool>();
        std::size_t const max_num_neighbors
            = vm["max-num-neighbors"].as<std::size_t>();

        std::string const graphfile = vm["graph"].as<std::string>();

        // Verify the input files are available for reading
        bool rc;
        rc = fexists(graphfile);
        if ( !rc ) {
          std::cerr << " File " << graphfile << " not found! Exiting... " << std::endl;
          hpx::finalize();
          return 0;
        }
        rc = fexists(searchfile);
        if ( !rc ) {
          std::cerr << " File " << searchfile << " not found! Exiting... " << std::endl;
          hpx::finalize();
          return 0;
        }

        // Read in the graph file -- timing not reported
        hpx::util::high_resolution_timer readtime;
        std::vector<std::size_t> nodelist,neighborlist;
        {
          std::string line;
          std::string val1,val2;
          std::ifstream myfile;
          myfile.open(graphfile);
          if (myfile.is_open()) {
            while (myfile.good()) {
              while (std::getline(myfile,line)) {
                std::istringstream isstream(line);
                std::getline(isstream,val1,' ');
                std::getline(isstream,val2,' ');
                std::size_t node = boost::lexical_cast<std::size_t>(val1);
                std::size_t neighbor = boost::lexical_cast<std::size_t>(val2);
                // increment all nodes and neighbors by 1; the smallest edge number is 1
                // edge 0 is reserved for the parent of the root and for unvisited edges
                nodelist.push_back(node+1);
                neighborlist.push_back(neighbor+1);
              }
            }
          }
        }

        // read in the searchfile containing the root vertices to search -- timing not reported
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
        std::cout << "Elapsed time during read: " << readtime.elapsed() << " [s]" << std::endl;

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
            hpx::components::get_component_type<bfs::server::point>();

        // ---------------------------------------------------------------
        // get a unique list of nodes -- this is what determines ne, the number of components
        boost::numeric::ublas::mapped_vector<std::size_t> index;
        for (std::size_t i=0;i<nodelist.size();i++) {
          if ( index.find_element(nodelist[i]) == 0)  index.insert_element( nodelist[i],1);
          if ( index.find_element(neighborlist[i]) == 0) index.insert_element( neighborlist[i],1);
        }
        // sum index
        std::size_t num_elements = index.nnz(); //std::accumulate( index.begin(), index.end(), 0);
        index.resize(nodelist.size());     // work around a bug in sparse_vector
        if ( num_elements%grainsize != 0 ) num_elements += grainsize-(num_elements%grainsize);
        std::size_t ne = num_elements/grainsize;

        // Decide where everyone should go
        std::size_t component_id = 0;
        std::size_t count = 0;
        for (boost::numeric::ublas::mapped_vector<std::size_t>::iterator it=index.begin();
                         it!=index.end();++it) {
          if ( *it != 0 ) {
            *it = component_id;
            count++;
          }
          if ( count == grainsize-1 ) {
            component_id++;
            count = 0;
          }
        }

        // ---------------------------------------------------------------

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
        // Put the graph in the data structure
        std::vector<hpx::lcos::future<void> > init_phase;

        for (std::size_t i=0;i<ne;i++) {
          init_phase.push_back(points[i].init_async(i,grainsize,max_num_neighbors,
                                                    nodelist,neighborlist,index,max_levels));
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation.
        hpx::lcos::wait(init_phase);
        double kernel1_time = kernel1time.elapsed();
        std::cout << "Elapsed time during kernel 1: " << kernel1_time << " [s]" << std::endl;

        ///////////////////////////////////////////////////////////////////////
        // KERNEL 2  --- TIMED
        std::vector<double> kernel2_time;
        std::vector<double> kernel2_nedge;
        kernel2_time.resize(searchroot.size());
        kernel2_nedge.resize(searchroot.size());
        hpx::util::high_resolution_timer kernel2time;

        bool validation = false;

        // go through each root position
        for (std::size_t step=0;step<searchroot.size();step++) {
          hpx::util::high_resolution_timer kernel2time;

          std::vector<hpx::lcos::future<std::vector<std::size_t> > > traverse_phase;

          // Traverse the graph.
          std::size_t level = 0;

          // Create the parent vectors.
          std::vector<std::vector<std::size_t> > parents;
          for (std::size_t i=0;i<max_levels;i++) {
            parents.push_back(std::vector<std::size_t>());
          }

          // TEST
          std::vector<hpx::lcos::future<std::vector<nodedata> > > depth_traverse_phase;
          std::vector<std::vector<nodedata> > result;
          depth_traverse_phase.push_back( points[ index(searchroot[step]) ].depth_traverse_async(level,searchroot[step],searchroot[step]) );
          hpx::lcos::wait(depth_traverse_phase,result);
          // END TEST
#if 0
          std::vector<std::vector<std::size_t> > neighbors,alt_neighbors;

          // Install the root node.
          parents[level].push_back( searchroot[step] );
          // identify the component which has the root
          traverse_phase.push_back( points[ index(searchroot[step]) ].traverse_async(level,searchroot[step],searchroot[step]) );

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
                std::size_t parent = parents[k-1][i];

                for (std::size_t j=0;j<neighbors[i].size();j++) {
                  parents[k].push_back( neighbors[i][j] );

                  // Create a future encapsulating an asynchronous call to
                  // the traverse action of bfs::point.
                  traverse_phase.push_back(points[ index(neighbors[i][j]) ].traverse_async(k,parent,neighbors[i][j]));
                }
              }

              // Wait for this phase to finish
              hpx::lcos::wait(traverse_phase,alt_neighbors);

            } else {
              // Clear the neighbor vector.
              neighbors.resize(0);

              for (std::size_t i=0;i<alt_neighbors.size();i++) {
                // Set the current parent.
                std::size_t parent = parents[k-1][i];

                for (std::size_t j=0;j<alt_neighbors[i].size();j++) {
                  parents[k].push_back( alt_neighbors[i][j] );

                  // Create a future encapsulating an asynchronous call to
                  // the traverse action of bfs::point.
                  traverse_phase.push_back(points[ index(alt_neighbors[i][j]) ].traverse_async(k,parent,alt_neighbors[i][j]));
                }
              }

              // Wait for this phase to finish
              hpx::lcos::wait(traverse_phase,neighbors);
            }
          }
#endif
          kernel2_time[step] = kernel2time.elapsed();

          if ( validator ) {
            // Validate  -- Not timed
            //std::cout << " Validating searchkey " << step << std::endl;
            // Get the parent of every edge
            std::vector<std::size_t> nodeparents;
            std::vector<std::size_t> nodelevels;
            std::vector<std::size_t> nodeparentsindex;
            for (boost::numeric::ublas::mapped_vector<std::size_t>::iterator it=index.begin();
                           it!=index.end();++it) {
              nodeparents.push_back(points[ *it ].get_parent(it.index()));
              nodelevels.push_back(points[ *it ].get_level(it.index()));
              nodeparentsindex.push_back(it.index());
            }

            validation = true;
            //std::cout << " Validating graph for " << searchroot[step] << std::endl;
            std::size_t num_edges;
            int rc = validate(nodeparents,nodelevels,nodeparentsindex,nodelist,
                              neighborlist,searchroot[step],num_edges);
            kernel2_nedge[step] = (double) num_edges;
            if ( rc <= 0 ) {
              validation = false;
              std::cout << " Validation failed for searchroot " << searchroot[step] << " rc " << rc << std::endl;
              break;
            }
            //std::cout << " Validation for " << searchroot[step] << " passed! " << std::endl;
          }

          // Reset for the next root
          std::vector<hpx::lcos::future<void> > reset_phase;
          for (std::size_t i=0;i<ne;i++) {
            reset_phase.push_back(points[i].reset_visited_async(i));
          }
          hpx::lcos::wait(reset_phase);
        }

        if ( validation && validator ) {
          // Prep output statistics
          double minimum,mean,stdev,firstquartile,median,thirdquartile,maximum;
          get_statistics(kernel2_time,minimum,mean,stdev,firstquartile,
                         median,thirdquartile,maximum);

          double n_min,n_mean,n_stdev,n_firstquartile,n_median,n_thirdquartile,n_maximum;
          get_statistics(kernel2_nedge,n_min,n_mean,n_stdev,n_firstquartile,
                         n_median,n_thirdquartile,n_maximum);

          // Print time statistics
          //std::cout << " SCALE: " << SCALE << std::endl;
          //std::cout << " edgefactor: " << edgefactor << std::endl;
          //std::cout << " NBFS: " << NBFS << std::endl;
          std::cout << " construction_time:  " << kernel1_time << std::endl;

          std::cout << " min_time:             " << minimum << std::endl;
          std::cout << " firstquartile_time:   " << firstquartile << std::endl;
          std::cout << " median_time:          " << median << std::endl;
          std::cout << " thirdquartile_time:   " << thirdquartile << std::endl;
          std::cout << " max_time:             " << maximum << std::endl;
          std::cout << " mean_time:            " << mean << std::endl;
          std::cout << " stddev_time:          " << stdev << std::endl;

          std::cout << " min_nedge:            " << n_min << std::endl;
          std::cout << " firstquartile_nedge:  " << n_firstquartile << std::endl;
          std::cout << " median_nedge:         " << n_median << std::endl;
          std::cout << " thirdquartile_nedge:  " << n_thirdquartile << std::endl;
          std::cout << " max_nedge:            " << n_maximum << std::endl;
          std::cout << " mean_nedge:           " << n_mean << std::endl;
          std::cout << " stddev_nedge:         " << n_stdev << std::endl;

          std::vector<double> TEPS;
          TEPS.resize(kernel2_nedge.size());
          for (std::size_t i=0;i<kernel2_nedge.size();i++) {
            TEPS[i] = kernel2_nedge[i]/kernel2_time[i];
          }
          std::size_t N = TEPS.size();
          double TEPS_min,TEPS_mean,TEPS_stdev,TEPS_firstquartile,TEPS_median,TEPS_thirdquartile,TEPS_maximum;
          get_statistics(TEPS,TEPS_min,TEPS_mean,TEPS_stdev,TEPS_firstquartile,
                         TEPS_median,TEPS_thirdquartile,TEPS_maximum);

          // Harmonic standard deviation from:
          // Nilan Norris, The Standard Errors of the Geometric and Harmonic
          // Means and Their Application to Index Numbers, 1940.
          // http://www.jstor.org/stable/2235723
          TEPS_stdev = TEPS_stdev/(TEPS_mean*TEPS_mean*sqrt( (double) (N-1) ) );

          std::cout << " min_TEPS:            " << TEPS_min << std::endl;
          std::cout << " firstquartile_TEPS:  " << TEPS_firstquartile << std::endl;
          std::cout << " median_TEPS:         " << TEPS_median << std::endl;
          std::cout << " thirdquartile_TEPS:  " << TEPS_thirdquartile << std::endl;
          std::cout << " max_TEPS:            " << TEPS_maximum << std::endl;
          std::cout << " harmonic_mean_TEPS:  " << TEPS_mean << std::endl;
          std::cout << " harmonic_stddev_TEPS:" << TEPS_stdev << std::endl;
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
        ("grainsize", value<std::size_t>()->default_value(10000),
            "the grainsize of the components")
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

