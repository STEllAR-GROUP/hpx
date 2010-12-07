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
        par->cfl          = 0.15;
        int nx0          = 33;
        par->nt0         = numsteps;
        par->minx0       =   0.0;
        par->maxx0       =  15.0;
        par->x1  =  -2.5;
        par->x2  =   3.5;
        par->x3  =   6.5;
        par->x4  =   12.5;
        par->id_x0          =  10.0;
        par->id_sigma       =  0.9;
        par->id_amp         =  0.0001;
        par->lambda         =  1.0;
        par->v         =  0.1;
        par->eps         =  0.0;
        par->output_level =  0;
        par->granularity =  3;

        int scheduler = 1;  // 0: global scheduler
                            // 1: parallel scheduler
        std::string parfile;
        if (vm.count("parfile")) {
            parfile = vm["parfile"].as<std::string>();
            hpx::util::section pars(parfile);

            if ( pars.has_section("cosmo_amr") ) {
              hpx::util::section *sec = pars.get_section("cosmo_amr");
              if ( sec->has_entry("lambda") ) {
                std::string tmp = sec->get_entry("lambda");
                par->lambda = atof(tmp.c_str());
              }
              if ( sec->has_entry("v") ) {
                std::string tmp = sec->get_entry("v");
                par->v = atof(tmp.c_str());
              }
              if ( sec->has_entry("cfl") ) {
                std::string tmp = sec->get_entry("cfl");
                par->cfl = atof(tmp.c_str());
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
              if ( sec->has_entry("id_amp") ) {
                std::string tmp = sec->get_entry("id_amp");
                par->id_amp = atof(tmp.c_str());
              }
              if ( sec->has_entry("id_x0") ) {
                std::string tmp = sec->get_entry("id_x0");
                par->id_x0 = atof(tmp.c_str());
              }
              if ( sec->has_entry("id_sigma") ) {
                std::string tmp = sec->get_entry("id_sigma");
                par->id_sigma = atof(tmp.c_str());
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
              if ( sec->has_entry("x1") ) {
                std::string tmp = sec->get_entry("x1");
                par->x1 = atof(tmp.c_str());
              }
              if ( sec->has_entry("x2") ) {
                std::string tmp = sec->get_entry("x2");
                par->x2 = atof(tmp.c_str());
              }
              if ( sec->has_entry("x3") ) {
                std::string tmp = sec->get_entry("x3");
                par->x3 = atof(tmp.c_str());
              }
              if ( sec->has_entry("x4") ) {
                std::string tmp = sec->get_entry("x4");
                par->x4 = atof(tmp.c_str());
              }
            }
        }
        
        // derived parameters
        if ( nx0%par->granularity != 0 ) {
          std::cerr << " PROBLEM : nx0 must be divisible by the granularity " << std::endl;
          std::cerr << " nx0 " << nx0 << " granularity " << par->granularity << std::endl;
          BOOST_ASSERT(false);
        }
        par->nx0 = nx0/par->granularity;

        par->nx[0] = par->nx0;
        
        BOOST_ASSERT(par->x2 > par->x1);

        // estimate how many px threads we will need to cover these two regions
        for (int dxstep=0;dxstep<2;dxstep++) 
        {
          par->level_begin.resize(0);
          par->level_end.resize(0);
          par->ghostpoint.resize(0);
          par->ghostpoint_LoR.resize(0);
          par->cghostpoint.resize(0);
          par->cghostpoint_LoR.resize(0);
          par->level_index.resize(0);
          par->offset.resize(0);
          par->rowsize.resize(0);
          par->level_row.resize(0);

          had_double_type est_dx0;
          // initial guess
          if ( dxstep == 0 ) {
            est_dx0 = (par->maxx0 - par->minx0)/(nx0-1);
          } else {
            est_dx0 = par->dx0;
          }

          had_double_type tt1 = (par->x2 - par->x1)/(est_dx0/pow(2.0,par->allowedl))/par->granularity;
          double t1 = tt1;
          int est_r1 = (int) t1;

          par->nx[par->allowedl] = est_r1;
          for (int i=par->allowedl-1;i>0;i--) {
            par->nx[i] = par->nx[i+1] + 6;
          }
          // the number of coarse px threads
          had_double_type cr1;
          cr1 = par->x1;
          for (int j=par->allowedl-1;j>0;j--) {
            cr1 += (5+par->allowedl)*par->granularity*est_dx0/pow(2.0,j);
          }

          had_double_type cr2;
          cr2 = par->x2;
          //for (int j=par->allowedl-1;j>0;j--) {
          //  cr2 += 3*par->granularity*est_dx0/pow(2.0,j);
          //}


          had_double_type d_t0 = (cr1-par->minx0)/est_dx0/par->granularity;
          had_double_type d_t1 = (par->maxx0 - cr2)/est_dx0/par->granularity;
          double s0 = d_t0;
          double s1 = d_t1;
          int c0 = (int) s0;
          int c1 = (int) s1;

          par->level_begin.push_back(0);
          par->level_end.push_back(c0-3*(par->allowedl-1));
          par->ghostpoint.push_back(c0-3*(par->allowedl-1));
          par->ghostpoint_LoR.push_back(0);
          par->cghostpoint.push_back(c0-3*(par->allowedl-1)-1);
          par->cghostpoint_LoR.push_back(0);
          par->level_index.push_back(0);
          par->offset.push_back(0);
          for (int j=1;j<par->allowedl;j++) {
            par->level_begin.push_back( c0-3*(par->allowedl-j) );
            par->level_end.push_back(c0 -3*(par->allowedl-j) + 3);
            par->ghostpoint.push_back(c0 -3*(par->allowedl-j) + 3);
            par->ghostpoint_LoR.push_back(0);
            par->cghostpoint.push_back(c0 -3*(par->allowedl-j) + 3-1);
            par->cghostpoint_LoR.push_back(0);
            par->level_index.push_back(j);
            par->offset.push_back(c0-3*(par->allowedl-j));
          }
          par->level_begin.push_back( c0 );
          par->level_end.push_back(c0 + est_r1);
          par->ghostpoint.push_back(c0 + est_r1-1);
          par->ghostpoint_LoR.push_back(1);
          par->cghostpoint.push_back(c0 + est_r1);
          par->cghostpoint_LoR.push_back(1);
          par->level_index.push_back(par->allowedl);
          par->offset.push_back(c0);
          for (int j=par->allowedl-1;j>0;j--) {
            par->level_begin.push_back(c0 + 3*(par->allowedl-1-j) + est_r1);
            par->level_end.push_back(c0 + 3*(par->allowedl-1-j) + 3 + est_r1);
            par->ghostpoint.push_back(c0 + 3*(par->allowedl-1-j) + 3 + est_r1-1);
            par->ghostpoint_LoR.push_back(1);
            par->cghostpoint.push_back(c0 + 3*(par->allowedl-1-j) + 3 + est_r1);
            par->cghostpoint_LoR.push_back(1);
            par->level_index.push_back(j);
          }
          par->level_begin.push_back( c0 + 3*(par->allowedl-1) + est_r1 );
          par->level_end.push_back(c0 + 3*(par->allowedl-1) + est_r1 + c1);
          par->level_index.push_back(0);

          par->nx[0] = c0 + 3*(par->allowedl-1) + est_r1 + c1;
          for (int j=0;j<=par->allowedl;j++) {
            par->rowsize.push_back(par->nx[j]);
          }

          int num_rows = (int) pow(2,par->allowedl);
          num_rows *= 2; // we take two timesteps in the mesh
          for (int i=0;i<num_rows;i++) {
            int level = -1;
            for (int j=par->allowedl;j>=0;j--) {
              int tmp = (int) pow(2,j); 
              if ( i%tmp == 0 ) {
                level = par->allowedl-j;
                par->level_row.push_back(level);
                break;
              }
            }
          }

          had_double_type np = 0.0;
          int left_level,oleft_level; 
          oleft_level = 0;
          for (int j=0;j<par->nx[0];j++) {
            for (int k=0;k<par->level_begin.size();k++) {
              if ( j >= par->level_begin[k] && j < par->level_end[k] ) {
                left_level = par->level_index[k];
                break;
              }
            }
            if ( left_level < oleft_level ) {
              np -= 1.0/pow(2.0,left_level);
            } 
            np += par->granularity/pow(2.0,left_level);
          }
          par->dx0 = (par->maxx0 - par->minx0)/(np-1);
          par->dt0 = par->cfl*par->dx0;
    
        }
        //std::cout << " TEST " << par->level_begin.size() << std::endl;
        //for (int k=0;k<par->level_begin.size();k++) {
        //  std::cout << " TEST " << par->level_begin[k] << " " << par->level_end[k] << std::endl;
        //}

        // figure out the number of points
        numvals = par->rowsize[0];

        //had_double_type tmp2 = 3*pow(2,par->allowedl);
        //int num_rows = (int) tmp2;
        //numsteps = numsteps*3 - 2;

        // create output file to append to
        FILE *fdata;
        fdata = fopen("phi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("Pi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("chi.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("a.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("f.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("g.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("b.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("q.dat","w");
        fprintf(fdata,"\n");
        fclose(fdata);

        fdata = fopen("r.dat","w");
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

