//  Copyright (c) 2007-2011 Matthew Anderson

#include<iostream>
#include<vector>
#include<math.h>
#include "fname.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "gtc_hpx/point.hpp"

#include <hpx/components/distributing_factory/distributing_factory.hpp>

extern "C" {void FNAME(gtc)(); }

bool fexists(std::string const filename)
{
  std::ifstream ifile(filename);
  return ifile ? true : false;
}

/// This function initializes a vector of \a gtc::point clients,
/// connecting them to components created with
/// \a hpx::components::distributing_factory.
inline void
init(hpx::components::server::distributing_factory::iterator_range_type const& r,
    std::vector<gtc::point>& p)
{
    BOOST_FOREACH(hpx::naming::id_type const& id, r)
    {
        p.push_back(gtc::point(id));
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
        std::string const parfilename = vm["file"].as<std::string>();

        // Check to make sure the file exists
        //std::string check = parfilename;
        //check.append(".dat");
     //   bool rc = fexists(parfilename); 
     //   if ( !rc ) {
     //     std::cerr << " Par file " << parfilename << " not found! Exiting... " << std::endl;
     //     hpx::finalize();
     //     return 0;
     //   } else {
          // The EPIC input parameter system is hard to follow.
          // Just reuse what they have
     //     char datastem[120]; 
     //     sprintf(datastem,"%s",parfilename.c_str());
     //     FNAME(gtc)(datastem);
     //   }
        std::cout << " HELLO WORLD " << std::endl;
        FNAME(gtc)();
        std::cout << " HELLO WORLD A" << std::endl;
#if 0
        // placeholder
        std::size_t const number_partitions = 2;
        std::size_t scale = 1;
        scale *= 1E8;

        // number_partitions defines the size of the partition
        // for Additive Schwarz to work, we will need more partitions 
        // than just number_partitions.  number_partitions should be as 
        // small as possible for performance reasons; however, it can't be too
        // small since the partitioned graph won't fit into memory if it is too small
        std::size_t num_pe = number_partitions; // actual number of partitions is num_pe
        std::cout << " Number of components: " << num_pe << std::endl;

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Create a distributing factory locally. The distributing factory can
        // be used to create blocks of components that are distributed across
        // all localities that support that component type.
        hpx::components::distributing_factory factory;
        factory.create(hpx::find_here());

        // Get the component type for our point component.
        hpx::components::component_type block_type =
            hpx::components::get_component_type<gtc::server::point>();

        // ---------------------------------------------------------------
        // Create ne point components with distributing factory.
        // These components will be evenly distributed among all available
        // localities supporting the component type.
        hpx::components::distributing_factory::result_type blocks =
            factory.create_components(block_type, num_pe);

        ///////////////////////////////////////////////////////////////////////
        // This vector will hold client classes referring to all of the
        // components we just created.
        std::vector<gtc::point> points;

        // Populate the client vectors.
        init(hpx::components::server::locality_results(blocks), points);

        // Begin Kernel 2
        hpx::util::high_resolution_timer kernel2time;
        {
          std::vector<hpx::lcos::future<void> > bfs_phase;
          for (std::size_t i=0;i<num_pe;i++) {
            bfs_phase.push_back(points[i].bfs_async(scale));
          }
          hpx::lcos::wait(bfs_phase);
        }
        double k2time = kernel2time.elapsed();

        std::cout << " Time: " << k2time << std::endl;
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
        ("file", value<std::string>()->default_value(
                "exp_flush.dat"));

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}

