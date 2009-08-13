//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>

#include "amr/stencil_value.hpp"
#include "amr/functional_component.hpp"
#include "amr/amr_mesh.hpp"
#include "amr_c/stencil.hpp"
#include "amr_c/logging.hpp"

#include "amr_c_test/rand.hpp"

namespace po = boost::program_options;

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(std::size_t numvals, std::size_t numsteps, std::size_t stencilsize, bool do_logging)
{
    std::cout << " main Stencilsize : " << stencilsize << std::endl;
    // get component types needed below
    components::component_type function_type = 
        components::get_component_type<components::amr::stencil>();
    components::component_type logging_type = 
        components::get_component_type<components::amr::server::logging>();

    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        components::amr::amr_mesh mesh (
            components::amr::amr_mesh::create(here, 1, true));

        hpx::util::high_resolution_timer t;
        std::vector<naming::id_type> result_data(
            mesh.init_execute(function_type, numvals, numsteps, stencilsize,
                do_logging ? logging_type : components::component_invalid));
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
            ("mean,M", po::value<double>(), 
                "mean value of specified distribution")
            ("stddev,S", po::value<double>(), 
                "variance value of specified distribution")
            ("numsteps,s", po::value<std::size_t>(), 
                "the number of time steps to use for the computation")
            ("stencilsize,z", po::value<std::size_t>(), 
                "the total (left and right) stencil size for a node")
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

        char pdist = 'u';
        if (vm.count("dist"))
            pdist = vm["dist"].as<std::string>()[0];

        double mean = 1.0;
        if (vm.count("mean"))
            mean = vm["mean"].as<double>();

        double stddev = 0.0;
        if (vm.count("stddev"))
            stddev = vm["stddev"].as<double>();

        if (vm.count("verbose"))
            do_logging = true;

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        std::size_t numvals = 3;
        if (vm.count("numvals"))
            numvals = vm["numvals"].as<std::size_t>();

        std::size_t numsteps = 3;
        if (vm.count("numsteps"))
            numsteps = vm["numsteps"].as<std::size_t>();
        
        std::size_t stencilsize = 3;
        if (vm.count("stencilsize"))
            stencilsize = vm["stencilsize"].as<std::size_t>();

        initrand(42, pdist, mean, stddev, numsteps, numvals, num_threads);

        // initialize and start the HPX runtime
        hpx::runtime rt(hpx_host, hpx_port, agas_host, agas_port, mode);
        if (mode == hpx::runtime::worker) 
            rt.run(num_threads);
        else 
            rt.run(boost::bind (hpx_main, numvals, numsteps, stencilsize, do_logging), num_threads);
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

