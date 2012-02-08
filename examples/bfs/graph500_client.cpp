//  Copyright (c) 2007-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include<iostream>
#include<vector>
#include<math.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "graph500/make_graph.h"
#include "graph500/point.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

static void compute_edge_range(int rank, int size, int64_t M, int64_t* start_idx, int64_t* end_idx) {
  int64_t rankc = (int64_t)(rank);
  int64_t sizec = (int64_t)(size);
  *start_idx = rankc * (M / sizec) + (rankc < (M % sizec) ? rankc : (M % sizec));
  *end_idx = (rankc + 1) * (M / sizec) + (rankc + 1 < (M % sizec) ? rankc + 1 : (M % sizec));
}

// this routine mirrors the matlab validation routine
int validate(std::vector<std::size_t> const& parent,
             std::vector<std::size_t> const& levels,
             std::vector<std::size_t> const& parentindex,
             std::vector<std::size_t> const& nodelist,
             std::vector<std::size_t> const& neighborlist,
             std::size_t searchkey,std::size_t &num_edges);

void clean_up(std::vector<nodedata> &full,std::size_t scale,std::vector<std::size_t> &parent);

void get_statistics(std::vector<double> const& x, double &minimum, double &mean, double &stdev, double &firstquartile,
                                                  double &median, double &thirdquartile, double &maximum);

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

        // number_partitions defines the size of the partition
        // for Additive Schwarz to work, we will need more partitions 
        // than just number_partitions.  number_partitions should be as 
        // small as possible for performance reasons; however, it can't be too
        // small since the partitioned graph won't fit into memory if it is too small
        std::size_t num_pe = number_partitions; // actual number of partitions is num_pe
        if ( number_partitions > 1 ) {
          // the additional paritions contain the additive Schwarz overlap 
          num_pe += 1;
        }
        std::cout << " Number of components: " << num_pe << std::endl;

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
            factory.create_components(block_type, num_pe);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<graph500::point> points;

        // Populate the client vectors.
        init(hpx::components::server::locality_results(blocks), points);

        ///////////////////////////////////////////////////////////////////////
        // Put the graph in the data structure
        std::vector<hpx::lcos::promise<void> > init_phase;

        for (std::size_t i=0;i<num_pe;i++) {
          init_phase.push_back(points[i].init_async(i,scale,number_partitions));
        }

        // We have to wait for the initialization to complete before we begin
        // the next phase of computation.
        hpx::lcos::wait(init_phase);
        double kernel1_time = kernel1time.elapsed();
        std::cout << "Elapsed time during kernel 1: " << kernel1_time << " [s]" << std::endl;
        // Generate the search roots
        std::vector<std::size_t> bfs_roots;
        bfs_roots.resize(1);  // the graph500 specifies 64 search roots
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
              // the smallest root is 1 ( 0 is reserved for unvisited edges )
              // increment by 1
              root++;
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
        std::vector<double> kernel2_nedge,kernel2_time;
        kernel2_nedge.resize(bfs_roots.size());
        kernel2_time.resize(bfs_roots.size());

        hpx::util::high_resolution_timer kernel2time;

        for (std::size_t j=0;j<bfs_roots.size();j++) {
          hpx::util::high_resolution_timer kernel2time;
          {
            std::vector<hpx::lcos::promise<void> > bfs_phase;
            for (std::size_t i=0;i<num_pe;i++) {
              bfs_phase.push_back(points[i].bfs_async(bfs_roots[j]));
            }
            hpx::lcos::wait(bfs_phase);
          }
          kernel2_time[j] = kernel2time.elapsed();

          // Validate
          // Return all nodes visited and their parents
          if ( validator ) {
            // This is an all-gather on the parents
            std::vector<hpx::lcos::promise< std::vector< nodedata > > > validate_phase;
            std::vector<std::vector<nodedata> > result;
            for (std::size_t i=0;i<num_pe;i++) {
              validate_phase.push_back(points[i].validate_async());
            }
            hpx::lcos::wait(validate_phase,result);

            // Prep prior to scatter to all components
            std::vector<nodedata> full;
            for (std::size_t i=0;i<num_pe;i++) {
              for (std::size_t j=0;j<result[i].size();j++) {
                full.push_back( result[i][j] );
              }
            }
            // remove duplicate nodes
            std::vector<std::size_t> parent;
            clean_up(full,scale,parent);

            // some validation can be done here
            if ( parent[bfs_roots[j]] != bfs_roots[j] ) {
              // the parent of the root is always itself;
              // if not, validation fails
              std::cerr << " Validation for root " << bfs_roots[j] << " fails; bfs_root parent " << parent[bfs_roots[j]] << std::endl;
            }

            // broadcast full parent list to the number_partition components for validation
            std::vector<validatedata> validate_result;
            std::vector<hpx::lcos::promise< validatedata > > scatter_phase;
            for (std::size_t i=0;i<number_partitions;i++) {
              scatter_phase.push_back(points[i].scatter_async(parent,bfs_roots[j],scale));
            }
            hpx::lcos::wait(scatter_phase,validate_result);

            // find the number of edges
            kernel2_nedge[j] = 0; 
            for (std::size_t i=0;i<validate_result.size();i++) {
              kernel2_nedge[j] +=  validate_result[i].num_edges;
              if ( validate_result[i].rc < 0 ) {
                std::cerr << " Validation failed for bfs_root " << bfs_roots[j] << " rc: " << validate_result[i].rc << std::endl;
              }
            }
          }

          // Reset
          {
            std::vector<hpx::lcos::promise<void> > reset_phase;
            for (std::size_t i=0;i<num_pe;i++) {
              reset_phase.push_back(points[i].reset_async());
            }
            hpx::lcos::wait(reset_phase);
          }
        }

        // get statistics
        if ( !validator ) {
          int64_t start_idx, end_idx;
          int64_t M = INT64_C(16) << scale;
          compute_edge_range(number_partitions-1, number_partitions, M, &start_idx, &end_idx);
          // needed for statistics -- this is an estimate unless validator is run
          for (std::size_t i=0;i<bfs_roots.size();i++) {
            kernel2_nedge[i] = end_idx; // this normally comes from the validation; it is an estimate here
          }
        }

        // Prep output statistics
        double minimum,mean,stdev,firstquartile,median,thirdquartile,maximum;
        get_statistics(kernel2_time,minimum,mean,stdev,firstquartile,
                       median,thirdquartile,maximum);

        double n_min,n_mean,n_stdev,n_firstquartile,n_median,n_thirdquartile,n_maximum;
        get_statistics(kernel2_nedge,n_min,n_mean,n_stdev,n_firstquartile,
                       n_median,n_thirdquartile,n_maximum);

        // Print time statistics
        std::cout << " SCALE: " << scale << std::endl;
        std::cout << " edgefactor: " << 16 << std::endl;
        std::cout << " NBFS: " << bfs_roots.size() << std::endl;
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
        ("validator", value<bool>()->default_value(false),
            "whether to run the validation (slow)");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

