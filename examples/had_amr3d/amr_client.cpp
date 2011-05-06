//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/util/asio_util.hpp>

#include <boost/fusion/include/at_c.hpp>
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

        // we are in spherical symmetry, r=0 is the smallest radial domain point
        components::amr::unigrid_mesh unigrid_mesh;
        unigrid_mesh.create(here);
        boost::shared_ptr<std::vector<naming::id_type> > result_data =
            unigrid_mesh.init_execute(function_type, numvals, numsteps,
                do_logging ? logging_type : components::component_invalid,par);

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl << std::flush;
 

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
        for (std::size_t i = 0; i < result_data->size(); ++i)
            components::stubs::memory_block::free((*result_data)[i]);
    }   // amr_mesh needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: had_amr3d_client [options]");
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
                "the number of operating system threads to spawn for this "
                "HPX locality")
            ("localities,l", po::value<int>(),
                "the number of localities to wait for at application startup "
                "(default is 1)")
            ("dist,d", po::value<std::string>(), 
                "random distribution type (uniform or normal)")
            ("random_ports", "use random ports for AGAS and parcels.")
            ("numsteps,s", po::value<std::size_t>(), 
                "the number of time steps to use for the computation")
            ("granularity,g", po::value<std::size_t>(), "the granularity of the data")
            ("dimensions,i", po::value<int>(), "the dimensions of the search space")
            ("refinement,e", po::value<std::size_t>(), "levels of refinement")
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
        std::cerr << "had_amr3d_client: exception caught: " << e.what() << std::endl;
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
        std::cerr << "had_amr3d_client: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "                  using default value instead: " << port << std::endl;
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
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> 
    global_runtime_type;
typedef hpx::runtime_impl<hpx::threads::policies::local_queue_scheduler> 
    local_runtime_type;
typedef hpx::runtime_impl<hpx::threads::policies::local_priority_queue_scheduler> 
    local_priority_runtime_type;

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
        int num_localities = 1;
        hpx::runtime_mode mode = hpx::runtime_mode_console;    // default is console mode
        bool do_logging = false;

        if (vm.count("random_ports") && !vm.count("agas") && !vm.count("hpx"))
        {
            using boost::fusion::at_c;

            boost::fusion::vector2<boost::uint16_t, boost::uint16_t> ports =
                hpx::util::get_random_ports();

            std::cout <<   "Randomized port for AGAS: " << at_c<0>(ports)
                      << "\nRandomized port for parcels: " << at_c<1>(ports)
                      << std::endl; 

            agas_port = at_c<0>(ports);
            hpx_port = at_c<1>(ports);
        }

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("localities"))
            num_localities = vm["localities"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime_mode_worker;

        if (vm.count("verbose"))
            do_logging = true;

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        std::size_t numvals;

        std::size_t numsteps = 3;
        if (vm.count("numsteps"))
            numsteps = vm["numsteps"].as<std::size_t>();
        
        std::size_t granularity = 50;
        if (vm.count("granularity"))
            granularity = vm["granularity"].as<std::size_t>();
        
        std::size_t allowedl = 0;
        if (vm.count("refinement"))
            allowedl = vm["refinement"].as<std::size_t>();
        
        int nx0 = 100;
        if (vm.count("dimensions"))
            nx0 = vm["dimensions"].as<int>();

        components::amr::Parameter par;

        // default pars
        par->allowedl    = allowedl;
        par->loglevel    = 2;
        par->output      = 1.0;
        par->output_stdout = 1;
        par->lambda      = 0.15;
        par->nt0         = numsteps;
        par->minx0       = -4.0;
        par->maxx0       =  4.0;
        par->ethreshold  =  0.005;
        par->R0          =  1.0;
        par->amp         =  0.0;
        par->amp_dot     =  1.0;
        par->delta       =  0.5;
        par->gw          =  5;
        par->buffer      =  3;
        par->eps         =  0.0;
        par->output_level =  0;
        par->granularity =  granularity;
        for (int i=0;i<maxlevels;i++) {
          // default
          par->refine_level[i] = 1.5;
        }

        int scheduler = 1;  // 0: global scheduler
                            // 1: parallel scheduler
                            // 2: parallel scheduler with priority queue
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
              if ( sec->has_entry("ghostwidth") ) {
                std::string tmp = sec->get_entry("ghostwidth");
                par->gw = atoi(tmp.c_str());
              }
              if ( sec->has_entry("buffer") ) {
                std::string tmp = sec->get_entry("buffer");
                par->buffer = atoi(tmp.c_str());
              }
              if ( sec->has_entry("eps") ) {
                std::string tmp = sec->get_entry("eps");
                par->eps = atof(tmp.c_str());
              }
              if ( sec->has_entry("granularity") ) {
                std::string tmp = sec->get_entry("granularity");
                par->granularity = atoi(tmp.c_str());
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
          par->nx[i] = nx0/par->granularity;
        }

        // for each row, record what the lowest level on the row is
        int num_rows = (int) pow(2.,par->allowedl);
        // account for prolongation and restriction (which is done every other step
        if ( par->allowedl > 0 ) {
          num_rows += (int) pow(2.,par->allowedl)/2;
        }
        num_rows *= 2; // we take two timesteps in the mesh
        if ( par->allowedl == 0 ) num_rows = 3;
        par->num_rows = num_rows;

        int ii = -1; 
        for (int i=0;i<num_rows;i++) {
          if (  (i+5)%3 != 0 ) {
            ii++;
          } 
          int level = -1;
          for (int j=par->allowedl;j>=0;j--) {
            int tmp = (int) pow(2.,j);
            if ( ii%tmp == 0 ) {
              level = par->allowedl-j;
              par->level_row.push_back(level);
              break;
            }
          }
        }

        // DEBUG
        //for (int i=0;i<num_rows;i++) {
        //  std::cout << " DEBUG level_row " << par->level_row[i] << std::endl;
        //}

        par->dx0 = (par->maxx0 - par->minx0)/(nx0-1);
        par->dt0 = par->lambda*par->dx0;

        par->min.resize(par->allowedl+1);
        par->max.resize(par->allowedl+1);
        par->min[0] = par->minx0;
        par->max[0] = par->maxx0;
        for (int i=1;i<=par->allowedl;i++) {
          par->min[i] = 0.5*par->min[i-1];
          par->max[i] = 0.5*par->max[i-1];
        }

        // DEBUG
        //for (int i=0;i<=par->allowedl;i++) {
        //  std::cout << " DEBUG " << par->min[i] << " " << par->max[i] << std::endl;
        //  std::cout << " DEBUG " << par->min[i] << " " << par->min[i]+((par->nx[i]-1)*par->granularity + par->granularity-1)*par->dx0/pow(2.0,i) << std::endl;
        //}

        par->rowsize.resize(par->allowedl+1);
        for (int i=0;i<=par->allowedl;i++) {
          par->rowsize[i] = 0;
          for (int j=par->allowedl;j>=i;j--) {
            par->rowsize[i] += par->nx[j]*par->nx[j]*par->nx[j];
          }
        }

        // figure out the number of points for row 0
        numvals = par->rowsize[0];
#if 0
        FILE *fdata;
        fdata = fopen("equator.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);
#endif

        // initialize and start the HPX runtime
        std::size_t executed_threads = 0;

        if (scheduler == 0) {
          global_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
          if (mode == hpx::runtime_mode_worker) 
              rt.run(num_threads,num_localities);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads,num_localities);

          executed_threads = rt.get_executed_threads();
        } 
        else if (scheduler == 1) {
          std::pair<std::size_t, std::size_t> init(/*vm["local"].as<int>()*/num_threads, 0);
          local_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
          if (mode == hpx::runtime_mode_worker) 
              rt.run(num_threads,num_localities);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads,num_localities);

          executed_threads = rt.get_executed_threads();
        } 
        else if (scheduler == 2) {
          std::pair<std::size_t, std::size_t> init(/*vm["local"].as<int>()*/num_threads, 0);
          local_priority_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
          if (mode == hpx::runtime_mode_worker) 
              rt.run(num_threads,num_localities);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads,num_localities);

          executed_threads = rt.get_executed_threads();
        } 
        else {
          BOOST_ASSERT(false);
        }

        std::cout << "Executed number of PX threads: " << executed_threads << std::endl;
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

