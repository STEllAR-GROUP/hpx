//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include "init_mpfr.hpp"
#include "amr/dynamic_stencil_value.hpp"
#include "amr/functional_component.hpp"
#include "amr/unigrid_mesh.hpp"
#include "amr_c/stencil.hpp"
#include "amr_c/logging.hpp"

#include "amr_c_test/rand.hpp"

namespace po = boost::program_options;

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
// initialize mpreal default precision
namespace hpx { namespace components { namespace amr 
{
    init_mpfr init_(true);
}}}

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
        
        // we are in spherical symmetry, r=0 is the smallest radial domain point             
        components::amr::unigrid_mesh unigrid_mesh;
        unigrid_mesh.create(here);
        result_data = unigrid_mesh.init_execute(function_type, numvals, numsteps,
            do_logging ? logging_type : components::component_invalid,par);
        printf("Elapsed time: %f s\n", t.elapsed());
    
    // provide some wait time to read the elapsed time measurement
    //std::cout << " Hit return " << std::endl;
    //int junk;
    //std::cin >> junk;

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

//         boost::this_thread::sleep(boost::posix_time::seconds(3)); 

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

        std::size_t numvals;

        std::size_t numsteps = 400;
        if (vm.count("numsteps"))
            numsteps = vm["numsteps"].as<std::size_t>();

        components::amr::Parameter par;

        // default pars
        par->allowedl    = 0;
        par->loglevel    = 2;
        par->output      = 1.0;
        par->output_stdout = 1;
        par->lambda      = 0.15;
        int nx0          = 33;
        par->nt0         = numsteps;
        par->minx0       = -15.0;
        par->maxx0       =  15.0;
        par->ethreshold  =  0.005;
        par->R0          =  8.0;
        par->amp         =  0.1;
        par->amp_dot     =  0.1;
        par->delta       =  1.0;
        par->PP          =  7;
        par->eps         =  0.0;
        par->output_level =  0;
        par->granularity =  3;
        for (int i=0;i<maxlevels;i++) {
          // default
          par->refine_level[i] = 1.5;
        }

        int scheduler = 1;  // 0: global scheduler
                            // 1: parallel scheduler
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
              if ( sec->has_entry("nx0") ) {
                std::string tmp = sec->get_entry("nx0");
                nx0 = atoi(tmp.c_str());
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
              if ( sec->has_entry("minx0") ) {
                std::string tmp = sec->get_entry("minx0");
                par->minx0 = atof(tmp.c_str());
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
              if ( sec->has_entry("amp_dot") ) {
                std::string tmp = sec->get_entry("amp_dot");
                par->amp_dot = atof(tmp.c_str());
              }
              if ( sec->has_entry("PP") ) {
                std::string tmp = sec->get_entry("PP");
                par->PP = atoi(tmp.c_str());
              }
              if ( sec->has_entry("eps") ) {
                std::string tmp = sec->get_entry("eps");
                par->eps = atof(tmp.c_str());
              }
              if ( sec->has_entry("granularity") ) {
                std::string tmp = sec->get_entry("granularity");
                par->granularity = atoi(tmp.c_str());
               // if ( par->granularity < 3 ) {
               //   std::cerr << " Problem: granularity must be at least 3 : " << par->granularity << std::endl;
               //   BOOST_ASSERT(false);
               // }
              }
              for (int i=0;i<par->allowedl;i++) {
                char tmpname[80];
                sprintf(tmpname,"refine_level_%d",i);
                if ( sec->has_entry(tmpname) ) {
                  std::string tmp = sec->get_entry(tmpname);
                  par->refine_level[i] = atof(tmp.c_str());
                }
              }

            }
        }

        
        // derived parameters
        if ( nx0%par->granularity != 0 ) {
          std::cerr << " PROBLEM : nx0 must be divisible by the granularity " << std::endl;
          std::cerr << " nx0 " << nx0 << " granularity " << par->granularity << std::endl;
          BOOST_ASSERT(false);
        }

        // step up refinement centered around the middle of the grid
        par->nx[0] = nx0/par->granularity;
        for (int i=1;i<par->allowedl+1;i++) {
          par->nx[i] = int(par->refine_level[i-1]*par->nx[i-1]);
          if ( par->nx[i]%2 == 0 ) par->nx[i] += 1;
          if ( par->nx[i] > 2*par->nx[i-1] - 5 ) par->nx[i] = 2*par->nx[i-1] - 5;
        }

        int num_rows = (int) pow(2.,par->allowedl);
        num_rows *= 2; // we take two timesteps in the mesh

        int ii = -1; 
        int level = 0;
        int count = 0;
        int count2 = 0; 
        int ct3 = 0;
        while (ct3 < num_rows ) {
          ii++;
          if ( count > 0 ) {
            if ( par->level_row[count-1] != par->allowedl &&
                 count2 < par->allowedl - par->level_row[count-1]
               ) {
              ii--; 
              count2++;
            } else {
              count2 = 0;
            } 
          }

          for (int j=par->allowedl;j>=0;j--) {
            int tmp = (int) pow(2.,j);
            if ( ii%tmp == 0 ) {
              level = par->allowedl-j;
              par->level_row.push_back(level);
              if ( count > 0 ) {
                if ( par->level_row[count-1] != par->level_row[count] ||
                     par->allowedl == 0 ) {
                  ct3++;
                } 
              } else {
                ct3++;
              }
              count++;
              break;
            }
          }
        }

        par->prores.resize(par->level_row.size());
        par->ndst.resize((par->allowedl+1)*par->level_row.size());
        // initialize
        for (int i=0;i<par->level_row.size();i++) {
          for (int j=0;j<par->allowedl+1;j++) {
            par->ndst[j + (par->allowedl+1)*i] = -1;
          }
        }
        if ( par->allowedl == 0 ) {
          par->prores[0] = 100;
          par->ndst[0] = 1;
        } else {
          for (int i=0;i<par->level_row.size()-1;i++) {
            if ( par->level_row[i] == par->level_row[i+1] ) {
              // Avoid race condition by staggering restriction, one level at a time
              if ( i == 0 ) {
                par->prores[i] = par->allowedl;
              } else {
                if ( par->prores[i-1] == 100 ) {
                  par->prores[i] = par->allowedl;
                } else {
                  par->prores[i] = par->prores[i-1]-1;
                }
              }

              // all levels go to the next row
              for (int j=par->level_row[i];j<par->allowedl+1;j++) {
                par->ndst[j + (par->allowedl+1)*i] = i+1;
              }
            } else {
              par->prores[i] = 100;
              int found;
              for (int j=par->level_row[i];j<par->allowedl+1;j++) {
                found = -1;
                // find the row containing a level equal to this or smaller
                for (int k=i+1;k<par->level_row.size();k++) {
                  if ( j >= par->level_row[k] ) {
                    par->ndst[j + (par->allowedl+1)*i] = k;
                    found = 1;
                    break;
                  }
                }
                if ( found != 1 ) {
                  par->ndst[j + (par->allowedl+1)*i] = 0;
                }
              }
            }
          }
        }
        par->prores[par->level_row.size()-1] = 100;
        par->ndst[par->allowedl + (par->allowedl+1)*(par->level_row.size()-1)] = 0;
         
        //for (int i=0;i<par->level_row.size();i++) {
        //  std::cout << " DEBUG " << i << " " << par->level_row[i] << " prores " << par->prores[i] << std::endl;
        //  std::cout << "                Destination: " << std::endl;
        //  for (int j=par->level_row[i];j<par->allowedl+1;j++) {
        //    std::cout << "                        level: " << j << " " << par->ndst[j+(par->allowedl+1)*i] << std::endl;
        //  }
        //}
        //BOOST_ASSERT(false);

        par->dx0 = (par->maxx0 - par->minx0)/(nx0-1);
        par->dt0 = par->lambda*par->dx0;

        par->min.resize(par->allowedl+1);
        par->min[0] = par->minx0;
        for (int i=par->allowedl;i>0;i--) {
          par->min[i] = 0.5*(par->maxx0 - par->minx0) + par->minx0  
                            - (par->nx[i]-1)/2 * par->dx0/pow(2.0,i) * par->granularity;
        }

        par->max.resize(par->allowedl+1);
        par->max[0] = par->maxx0;
        for (int i=par->allowedl;i>0;i--) {
          par->max[i] = par->min[i] + ((par->nx[i]-1)*par->granularity + par->granularity-1)*par->dx0/pow(2.0,i);
        }

        par->rowsize.resize(par->allowedl+1);
        for (int i=0;i<=par->allowedl;i++) {
          par->rowsize[i] = 0;
          for (int j=par->allowedl;j>=i;j--) {
            par->rowsize[i] += par->nx[j]*par->nx[j]*par->nx[j];
          }
        }

        //ghostwidth
        par->gw = 5;

        // figure out the number of points for row 0
        numvals = par->rowsize[0];

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

