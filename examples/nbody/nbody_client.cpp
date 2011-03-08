//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>



#include <hpx/hpx.hpp>


#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>

//#include "nbody/stencil_value.hpp"
#include "init_mpfr.hpp"
#include "nbody/dynamic_stencil_value.hpp"
#include "nbody/functional_component.hpp"
#include "nbody/unigrid_mesh.hpp"
#include "nbody_c/stencil.hpp"
#include "nbody_c/logging.hpp"

#include "nbody_c_test/rand.hpp"

namespace po = boost::program_options;

typedef struct {
    double dtime;
    double eps;
    double tolerance;
    double half_dt;
    double softening_2;
    double inv_tolerance_2;
    int iter;
    int num_bodies;
    int num_iterations;
} crit_vals;

typedef struct {
    double mass;
    double px, py, pz;
    double vx, vy, vz;
    double ax, ay, az;
} body;

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
// initialize mpreal default precision
namespace hpx { namespace components { namespace nbody 
{
 //   init_mpfr init_(true);
}}}


static inline void computeRootPos(const int num_bodies, double &box_dim, double center_position[],hpx::memory::default_vector<body>::type bodies) 
{
    double minPos[3];
    minPos[0] = 1.0E90; 
    minPos[1] = 1.0E90;
    minPos[2] = 1.0E90;
    double maxPos[3];
    maxPos[0] = -1.0E90;
    maxPos[1] = -1.0E90;
    maxPos[2] = -1.0E90;
   
    for (int i = 0; i < bodies.size(); ++i)
    {
       
        if (minPos[0] > bodies[i].px)
            minPos[0] = bodies[i].px;
        if (minPos[1] > bodies[i].py)
            minPos[1] = bodies[i].py;
        if (minPos[2] > bodies[i].pz)
            minPos[2] = bodies[i].pz;
        if (maxPos[0] < bodies[i].px)
            maxPos[0] = bodies[i].px;
        if (maxPos[1] < bodies[i].py)
            maxPos[1] = bodies[i].py;
        if (maxPos[2] < bodies[i].pz)
            maxPos[2] = bodies[i].pz;
    }
    box_dim = maxPos[0] - minPos[0];
    if (box_dim < (maxPos[1] - minPos[1]))
        box_dim = maxPos[1] - minPos[1];
    if (box_dim < (maxPos[2] - minPos[2]))
        box_dim = maxPos[2] - minPos[2];
    center_position[0] = (maxPos[0] + minPos[0]) / 2;
    center_position[1] = (maxPos[1] + minPos[1]) / 2;
    center_position[2] = (maxPos[2] + minPos[2]) / 2;
}



///////////////////////////////////////////////////////////////////////////////
int hpx_main(std::size_t numvals, std::size_t numsteps,bool do_logging,
             components::nbody::Parameter const& par)
{
    std::string input_file;
    crit_vals cv;

    // get component types needed below
    components::component_type function_type = 
        components::get_component_type<components::nbody::stencil>();
    components::component_type logging_type = 
        components::get_component_type<components::nbody::server::logging>();

    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();

        if ( par->loglevel > 0 ) {
          // over-ride a false command line argument
          do_logging = true;
        }

        hpx::util::high_resolution_timer t;

        components::nbody::unigrid_mesh unigrid_mesh;
        unigrid_mesh.create(here);
        
        input_file = par->input_file;
        if(input_file.size() == 0)
        {
            //hpx::finalize();
            return 0;
        }
            
        std::ifstream infile;
        infile.open(input_file.c_str());
        if (!infile)                                /* if there is a problem opening file */
        {                                           /* exit gracefully */
            std::cerr << "Can't open input file " << input_file << std::endl;
            exit (1);
        }
        infile >> cv.num_bodies;                   
        infile >> cv.num_iterations;               
        infile >> cv.dtime;                        
        infile >> cv.eps;                          
        infile >> cv.tolerance;                  
        cv.half_dt = 0.5 * cv.dtime;
        cv.softening_2 = cv.eps * cv.eps;
        cv.inv_tolerance_2 = 1.0 / (cv.tolerance * cv.tolerance);    
        
        std::cout << "Num Bodies " << cv.num_bodies << std::endl;
        hpx::memory::default_vector<body>::type bodies;
        hpx::memory::default_vector<body>::type::iterator bod_iter;
        bodies.resize(cv.num_bodies);
        
        for (int i =0; i < cv.num_bodies ; ++i)
        {
            double dat[7] = {0,0,0,0,0,0,0};
            infile >> dat[0] >> dat[1] >> dat[2] >> dat[3] >> dat[4] >> dat[5] >> dat[6];
            bodies[i].mass = dat[0];
            bodies[i].px = dat[1];
            bodies[i].py = dat[2];
            bodies[i].pz = dat[3];
            bodies[i].vx = dat[4];
            bodies[i].vy = dat[5];
            bodies[i].vz = dat[6];
        }
        
        for (int i =0; i < cv.num_bodies ; ++i)
        {
            std::cout << "body : "<< i << " : " << bodies[i].mass << " : " <<
            bodies[i].px << " : " << bodies[i].py << " : " << bodies[i].pz << std::endl;
            std::cout <<"           " << " : " << bodies[i].vx << " : " << bodies[i].vy 
            << " : " << bodies[i].vz << std::endl;
        }

        infile.close();  
        for (cv.iter = 0; cv.iter < cv.num_iterations; ++cv.iter)
        {
            double box_size, cPos[3];
            computeRootPos(cv.num_bodies, box_size, cPos, bodies);
            std::cout << "Center Position : " << cPos[0] << " " << cPos[1] << " " << cPos[2] << std::endl;
        }

        // for loop for iteration

        // recompute tree/mass
        
        std::vector<naming::id_type> result_data =
            unigrid_mesh.init_execute(function_type, numvals, numsteps,
                do_logging ? logging_type : components::component_invalid,par);

        // end for loop

        std::cout << "Elapsed time: " << t.elapsed() << " [s]" << std::endl;

        for (std::size_t i = 0; i < result_data.size(); ++i)
            components::stubs::memory_block::free(result_data[i]);
    }   // nbody_mesh needs to go out of scope before shutdown

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage:nbody_client [options]");
        desc_cmdline.add_options()
            ("input_file,i", po::value<std::string>(), 
            "Input File")
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
        std::cerr << "nbody_client: exception caught: " << e.what() << std::endl;
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
        std::cerr << "nbody_client: illegal port number given: " << v.substr(p+1) << std::endl;
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

        components::nbody::Parameter par;

        // default pars
        par->loglevel    = 2;
        par->output      = 1.0;
        par->output_stdout = 1;
        par->rowsize =  4;
        par->input_file="5_file";

        int scheduler = 1;  // 0: global scheduler
                            // 1: parallel scheduler
        std::string parfile;
        if (vm.count("parfile")) {
            parfile = vm["parfile"].as<std::string>();
            hpx::util::section pars(parfile);

            if ( pars.has_section("nbody") ) {
              hpx::util::section *sec = pars.get_section("nbody");
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
              if ( sec->has_entry("thread_scheduler") ) {
                std::string tmp = sec->get_entry("thread_scheduler");
                scheduler = atoi(tmp.c_str());
                BOOST_ASSERT( scheduler == 0 || scheduler == 1 );
              }
              if ( sec->has_entry("input_file") ) {
                std::string tmp = sec->get_entry("input_file");
                par->input_file = tmp;
              }
            }
        }

        
        // figure out the number of points for row 0
        numvals = par->rowsize;

        // initialize and start the HPX runtime
        std::size_t executed_threads = 0;

        int numsteps = 1;

        if (scheduler == 0) {
          global_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
          if (mode == hpx::runtime::worker) 
              rt.run(num_threads);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads);

          executed_threads = rt.get_executed_threads();
        } 
        else if (scheduler == 1) {
          std::pair<std::size_t, std::size_t> init(/*vm["local"].as<int>()*/num_threads, 0);
          local_runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode, init);
          if (mode == hpx::runtime::worker) 
              rt.run(num_threads);
          else 
              rt.run(boost::bind(hpx_main, numvals, numsteps, do_logging, par), num_threads);

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

