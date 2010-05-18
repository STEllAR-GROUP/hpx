//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>

//#include "amr/stencil_value.hpp"
#include "amr/dynamic_stencil_value.hpp"
#include "amr/functional_component.hpp"
#include "amr/unigrid_mesh.hpp"
#include "amr_c/stencil.hpp"
#include "amr_c/logging.hpp"

#include "amr_c_test/rand.hpp"

namespace po = boost::program_options;

using namespace hpx;

// prep_ports {{{
    void prep_ports(Array3D &dst_port,Array3D &dst_src,
                             Array3D &dst_step,Array3D &dst_size,Array3D &src_size)
    {
      int i,j;
      int counter;

      // vcolumn is the destination column number
      // vstep is the destination step (or row) number
      // vsrc_column is the source column number
      // vsrc_step is the source step number
      // vport is the output port number; increases consecutively
      std::vector<int> vcolumn,vstep,vsrc_column,vsrc_step,vport;

      // connect outputs for the zeroth row (the zeroth row outputs to the first row *and* the third row)
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(1);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(1);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(1);vcolumn.push_back(1);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(0);vstep.push_back(3);vcolumn.push_back(1);vport.push_back(3);

      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(1);vcolumn.push_back(2);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(3);vcolumn.push_back(1);vport.push_back(3);
      vsrc_step.push_back(0);vsrc_column.push_back(1);vstep.push_back(3);vcolumn.push_back(2);vport.push_back(4);

      i = 2;
      counter = 0;
      for (j=i-3;j<i+2;j++) {
        if ( j >= 0 && j <= 14 ) {
          vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(j);vport.push_back(counter);
          counter++;
        }
      }
      i = 2;
      counter = 4;
      for (j=i-7;j<i+2;j++) {
        if ( j > 0 && j < 10 ) {
          vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
          counter++;
        }
      }

      for (i=3;i<14;i++) {
        counter = 0;
        for (j=i-3;j<i+2;j++) {
          if ( j >= 0 && j <= 14 ) {
            vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      for (i=3;i<14;i++) {
        counter = 5;  // counter starts at 5 because this the first three output ports were already used in the lines above
        for (j=i-7;j<i+2;j++) {
          if ( j > 0 && j < 10 ) {
            vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      i = 14;
      counter = 0;
      for (j=i-3;j<i+2;j++) {
        if ( j >= 0 && j <= 14 ) {
          vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(1);vcolumn.push_back(j);vport.push_back(counter);
          counter++;
        }
      }
      i = 14;
      counter = 4;
      for (j=i-7;j<i+2;j++) {
        if ( j > 0 && j < 10 ) {
          vsrc_step.push_back(0);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
          counter++;
        }
      }

      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(1);vcolumn.push_back(12);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(1);vcolumn.push_back(13);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(1);vcolumn.push_back(14);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(3);vcolumn.push_back(8);vport.push_back(3);
      vsrc_step.push_back(0);vsrc_column.push_back(15);vstep.push_back(3);vcolumn.push_back(9);vport.push_back(4);

      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(1);vcolumn.push_back(13);vport.push_back(0);
      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(1);vcolumn.push_back(14);vport.push_back(1);
      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(1);vcolumn.push_back(14);vport.push_back(2);
      vsrc_step.push_back(0);vsrc_column.push_back(16);vstep.push_back(3);vcolumn.push_back(9);vport.push_back(3);

      // connect outputs for the first row (the first row only outputs to the second row)
      i = 0;
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(1);vport.push_back(2);

      for (i=1;i<14;i++) {
        counter = 0;  
        for (j=i-3;j<i+2;j++) {
          if ( j >= 0 && j <= 12 ) {
            vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      i = 14;
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(11);vport.push_back(0);
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(12);vport.push_back(1);
      vsrc_step.push_back(1);vsrc_column.push_back(i);vstep.push_back(2);vcolumn.push_back(12);vport.push_back(2);

      // connect outputs for the second row (the second row only outputs to the third row)
      i = 0;
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(1);vport.push_back(2);
    
      for (i=1;i<12;i++) {
        counter = 0;  
        for (j=i-3;j<i+2;j++) {
          if ( j >= 0 && j <= 10 ) {
            vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      i = 12;
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(9);vport.push_back(0);
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(10);vport.push_back(1);
      vsrc_step.push_back(2);vsrc_column.push_back(i);vstep.push_back(3);vcolumn.push_back(10);vport.push_back(2);

      // connect outputs for the third row (the third row outputs to the fourth row *and* the sixth row)
      vsrc_step.push_back(3);vsrc_column.push_back(0);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(0);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(1);

      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(4);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(1);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(3);

      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(4);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(4);vcolumn.push_back(2);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(3);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(4);
      vsrc_step.push_back(3);vsrc_column.push_back(2);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(5);

      i = 3;
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(1);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(2);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(3);vport.push_back(3);

      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(4);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(5);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(6);

      for (i=4;i<7;i++) {
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-4);vport.push_back(0);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-3);vport.push_back(1);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-2);vport.push_back(2);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-1);vport.push_back(3);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i);vport.push_back(4);

        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(5);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(6);
        vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(7);
      }

      i = 7;
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-4);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-3);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-2);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(4);vcolumn.push_back(i-1);vport.push_back(3);

      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(4);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(5);
      vsrc_step.push_back(3);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(6);

      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(4);vcolumn.push_back(4);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(4);vcolumn.push_back(5);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(4);vcolumn.push_back(6);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(3);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(4);
      vsrc_step.push_back(3);vsrc_column.push_back(8);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(5);

      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(4);vcolumn.push_back(5);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(4);vcolumn.push_back(6);vport.push_back(1);
      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(2);
      vsrc_step.push_back(3);vsrc_column.push_back(9);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(3);

      vsrc_step.push_back(3);vsrc_column.push_back(10);vstep.push_back(4);vcolumn.push_back(6);vport.push_back(0);
      vsrc_step.push_back(3);vsrc_column.push_back(10);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(1);

      // connect outputs for the fourth row (the fourth row only outputs to the fifth row)
      i = 0;
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(1);vport.push_back(2);

      for (i=1;i<6;i++) {
        counter = 0;  
        for (j=i-3;j<i+2;j++) {
          if ( j >= 0 && j <= 4 ) {
            vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }

      i = 6;
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(3);vport.push_back(0);
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(4);vport.push_back(1);
      vsrc_step.push_back(4);vsrc_column.push_back(i);vstep.push_back(5);vcolumn.push_back(4);vport.push_back(2);

      // connect outputs for the fifth row (the fifth row only outputs to the sixth row)
      i = 0;
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(0);
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(0);vport.push_back(1);
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(2);

      for (i=1;i<4;i++) {
        counter = 0;  
        for (j=i-3;j<i+2;j++) {
          if ( j >= 0 && j <= 2 ) {
            vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(j);vport.push_back(counter);
            counter++;
          }
        }
      }
      i = 4;
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(1);vport.push_back(0);
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(1);
      vsrc_step.push_back(5);vsrc_column.push_back(i);vstep.push_back(6);vcolumn.push_back(2);vport.push_back(2);

      // connect outputs for the sixth row (the sixth row only outputs to the zeroth row)
      for (i=0;i<6;i++) {
        vsrc_step.push_back(6);vsrc_column.push_back(0);vstep.push_back(0);vcolumn.push_back(i);vport.push_back(i);
        if ( i < 5 ) { 
          vsrc_step.push_back(6);vsrc_column.push_back(1);vstep.push_back(0);vcolumn.push_back(i+6);vport.push_back(i);
        }
        vsrc_step.push_back(6);vsrc_column.push_back(2);vstep.push_back(0);vcolumn.push_back(i+11);vport.push_back(i);
      }

      // Create a ragged 3D array
      for (j=0;j<vsrc_step.size();j++) {
        int column,step,src_column,src_step,port;
        src_column = vsrc_column[j]; src_step = vsrc_step[j];
        column = vcolumn[j]; step = vstep[j];
        port = vport[j];
        dst_port( step,column,dst_size(step,column,0) ) = port;
        dst_src(  step,column,dst_size(step,column,0) ) = src_column;
        dst_step( step,column,dst_size(step,column,0) ) = src_step;
        dst_size(step,column,0) += 1;
        src_size(src_step,src_column,0) += 1;
      }
       
      // sort the src step (or row) in descending order
      int t1,k,kk;
      int step,column;
      for (j=0;j<vsrc_step.size();j++) {
        step = vstep[j];
        column = vcolumn[j];
        for (kk=dst_size(step,column,0);kk>=0;kk--) {
          for (k=0;k<kk-1;k++) {
            if (dst_step( step,column,k) < dst_step( step,column,k+1) ) {
              // swap
              t1 = dst_step( step,column,k);
              dst_step( step,column,k) = dst_step( step,column,k+1);
              dst_step( step,column,k+1) = t1;
  
              // swap the src, port info too
              t1 = dst_src( step,column,k);
              dst_src( step,column,k) = dst_src( step,column,k+1);
              dst_src( step,column,k+1) = t1;
  
              t1 = dst_port( step,column,k);
              dst_port( step,column,k) = dst_port( step,column,k+1);
              dst_port( step,column,k+1) = t1;
            } else if ( dst_step( step,column,k) == dst_step( step,column,k+1) ) {
              //sort the src column in ascending order if the step is the same
              if (dst_src( step,column,k) > dst_src( step,column,k+1) ) {
                t1 = dst_src( step,column,k);
                dst_src( step,column,k) = dst_src( step,column,k+1);
                dst_src( step,column,k+1) = t1;

                // swap the src, port info too
                t1 = dst_port( step,column,k);
                dst_port( step,column,k) = dst_port( step,column,k+1);
                dst_port( step,column,k+1) = t1;
              }

            }
          }
        }
      }


    }
// }}}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(std::size_t numvals, std::size_t numsteps,bool do_logging,
             components::amr::Parameter const& par)
{
    // get component types needed below
    components::component_type function_type = 
        components::get_component_type<components::amr::stencil>();
    components::component_type logging_type = 
        components::get_component_type<components::amr::server::logging>();

    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();

        if ( par->loglevel > 0 ) {
          // over-ride a false command line argument
          do_logging = true;
        }

        hpx::util::high_resolution_timer t;
        std::vector<naming::id_type> result_data;
        
        if ( par->integrator == 1 ) {
            components::amr::unigrid_mesh unigrid_mesh;
            unigrid_mesh.create(here);
            result_data = unigrid_mesh.init_execute(function_type, numvals, numsteps,
                do_logging ? logging_type : components::component_invalid, par);
        } else {
          BOOST_ASSERT(false);
        }
        printf("Elapsed time: %f s\n", t.elapsed());

        // get some output memory_block_data instances
        /*
        std::cout << "Results: " << std::endl;
        for (std::size_t i = 0; i < result_data.size(); ++i)
        {
            components::access_memory_block<components::amr::stencil_data> val(
                components::stubs::memory_block::get(result_data[i]));
            std::cout << i << ": " << val->value_ << std::endl;
        }
        */

        boost::this_thread::sleep(boost::posix_time::seconds(3)); 

        for (std::size_t i = 0; i < result_data.size(); ++i)
            components::stubs::memory_block::free(result_data[i]);
    }   // amr_mesh needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: hpx_runtime [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("worker,w", "run this instance in worker (non-console) mode")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("numvals,n", po::value<std::size_t>(), 
                "the number of data points to use for the computation")
            ("dist,d", po::value<std::string>(), 
                "random distribution type (uniform or normal)")
            ("numsteps,s", po::value<std::size_t>(), 
                "the number of time steps to use for the computation")
            ("parfile,p", po::value<std::string>(), 
                "the parameter file")
            ("verbose,v", "print calculated values after each time step")
        ;

        po::store(po::command_line_parser(argc, argv)
            .options(desc_cmdline).run(), vm);
        po::notify(vm);

        // print help screen
        if (vm.count("help")) {
            std::cout << desc_cmdline;
            return false;
        }
    }
    catch (std::exception const& e) {
        std::cerr << "amr_client: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    std::string::size_type p = v.find_first_of(":");
    try {
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        std::cerr << "amr_client: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "            using default value instead: " << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// helper class for AGAS server initialization
class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
      : agas_pool_(), agas_(agas_pool_, host, port)
    {
        agas_.run(false);
    }
    ~agas_server_helper()
    {
        agas_.stop();
    }

private:
    hpx::util::io_service_pool agas_pool_; 
    hpx::naming::resolver_server agas_;
};

///////////////////////////////////////////////////////////////////////////////
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> global_runtime_type;
typedef hpx::runtime_impl<hpx::threads::policies::local_queue_scheduler> local_runtime_type;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
#if defined(MPFR_FOUND)
    mpfr::mpreal::set_default_prec(128);
#endif

    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline(argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode
        bool do_logging = false;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime::worker;

        if (vm.count("verbose"))
            do_logging = true;

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        std::size_t numvals = 8;
        if (vm.count("numvals"))
            numvals = vm["numvals"].as<std::size_t>();

        std::size_t numsteps = 3;
        if (vm.count("numsteps"))
            numsteps = vm["numsteps"].as<std::size_t>();

        components::amr::Parameter par;

        // default pars
        par->stencilsize = 7;
        par->integrator  = 1;
        par->allowedl    = 0;
        par->loglevel    = 0;
        par->output      = 1.0;
        par->output_stdout = 1;
        par->lambda      = 0.15;
        par->nx0         = numvals;
        par->nt0         = numsteps;
        par->minx0       =   0.0;
        par->maxx0       =  10.0;
        par->ethreshold  =  0.1;
        par->R0          =  8.0;
        par->amp         =  0.1;
        par->delta       =  1.0;
        par->PP          =  7;
        par->eps         =  0.0;
        par->fmr_radius  =  -999.0;
        par->output_level =  0;

        par->linearbounds = 1;
        int scheduler = 0;  // default: global scheduler

        std::string parfile;
        if (vm.count("parfile")) {
            parfile = vm["parfile"].as<std::string>();
            hpx::util::section pars(parfile);

            if ( pars.has_section("had_amr") ) {
              hpx::util::section *sec = pars.get_section("had_amr");
              if ( sec->has_entry("lambda") ) {
                std::string tmp = sec->get_entry("lambda");
                par->lambda = atof(tmp.c_str());
              }
              if ( sec->has_entry("allowedl") ) {
                std::string tmp = sec->get_entry("allowedl");
                par->allowedl = atoi(tmp.c_str());
              }
              if ( sec->has_entry("loglevel") ) {
                std::string tmp = sec->get_entry("loglevel");
                par->loglevel = atoi(tmp.c_str());
              }
              if ( sec->has_entry("output") ) {
                std::string tmp = sec->get_entry("output");
                par->output = atof(tmp.c_str());
              }
              if ( sec->has_entry("output_stdout") ) {
                std::string tmp = sec->get_entry("output_stdout");
                par->output_stdout = atoi(tmp.c_str());
              }
              if ( sec->has_entry("output_level") ) {
                std::string tmp = sec->get_entry("output_level");
                par->output_level = atoi(tmp.c_str());
              }
              if ( sec->has_entry("stencilsize") ) {
                std::string tmp = sec->get_entry("stencilsize");
                par->stencilsize = atoi(tmp.c_str());
              }
              if ( sec->has_entry("integrator") ) {
                std::string tmp = sec->get_entry("integrator");
                par->integrator = atoi(tmp.c_str());
                if ( par->integrator < 0 || par->integrator > 1 ) BOOST_ASSERT(false); 
              }
              if ( sec->has_entry("linearbounds") ) {
                std::string tmp = sec->get_entry("linearbounds");
                par->linearbounds = atoi(tmp.c_str());
              }
              if ( sec->has_entry("nx0") ) {
                std::string tmp = sec->get_entry("nx0");
                par->nx0 = atoi(tmp.c_str());
                // over-ride command line argument if present
                numvals = par->nx0;
                if ( par->nx0 < 16 ) {
                  std::cout << " Problem: you need to increase nx0 to at least 16 !" << std::endl;
                  BOOST_ASSERT(false);
                }
              }
              if ( sec->has_entry("nt0") ) {
                std::string tmp = sec->get_entry("nt0");
                par->nt0 = atoi(tmp.c_str());
                // over-ride command line argument if present
                numsteps = par->nt0;
              }
              if ( sec->has_entry("thread_scheduler") ) {
                std::string tmp = sec->get_entry("thread_scheduler");
                scheduler = atoi(tmp.c_str());
                BOOST_ASSERT( scheduler == 0 || scheduler == 1 );
              }
              if ( sec->has_entry("maxx0") ) {
                std::string tmp = sec->get_entry("maxx0");
                par->maxx0 = atof(tmp.c_str());
              }
              if ( sec->has_entry("ethreshold") ) {
                std::string tmp = sec->get_entry("ethreshold");
                par->ethreshold = atof(tmp.c_str());
              }
              if ( sec->has_entry("R0") ) {
                std::string tmp = sec->get_entry("R0");
                par->R0 = atof(tmp.c_str());
              }
              if ( sec->has_entry("delta") ) {
                std::string tmp = sec->get_entry("delta");
                par->delta = atof(tmp.c_str());
              }
              if ( sec->has_entry("amp") ) {
                std::string tmp = sec->get_entry("amp");
                par->amp = atof(tmp.c_str());
              }
              if ( sec->has_entry("PP") ) {
                std::string tmp = sec->get_entry("PP");
                par->PP = atoi(tmp.c_str());
              }
              if ( sec->has_entry("eps") ) {
                std::string tmp = sec->get_entry("eps");
                par->eps = atof(tmp.c_str());
              }
              if ( sec->has_entry("fmr_radius") ) {
                std::string tmp = sec->get_entry("fmr_radius");
                par->fmr_radius = atof(tmp.c_str());
              }
            }
        }

        // prep amr ports
        prep_ports(par->dst_port,par->dst_src,par->dst_step,par->dst_size,par->src_size);

        // derived parameters
        par->dx0 = (par->maxx0 - par->minx0)/(par->nx0-1);
        par->dt0 = par->lambda*par->dx0;

        // The stencilsize needs to be odd
        BOOST_ASSERT(par->stencilsize%2 != 0 );

        if ( par->integrator == 1 ) {
          numsteps *= 6;  // six subcycles each step
        }

        // create output file to append to
        FILE *fdata;
        fdata = fopen("chi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("Phi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("Pi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("energy.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("logcode1.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("logcode2.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        // initialize and start the HPX runtime
        if (scheduler == 0) {
          global_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
          if (mode == hpx::runtime::worker) 
              rt.run(num_threads);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads);
        } else if ( scheduler == 1) {
          std::pair<std::size_t, std::size_t> init(/*vm["local"].as<int>()*/num_threads, 0);
          local_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
          if (mode == hpx::runtime::worker) 
              rt.run(num_threads);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads);
        } else {
          BOOST_ASSERT(false);
        }
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }

    return 0;
}

