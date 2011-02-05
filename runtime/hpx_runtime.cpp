//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <memory>
#include <hpx/hpx.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(char const* name, int argc, char *argv[], 
    po::variables_map& vm)
{
    try {
        std::string usage("Usage: ");
        usage += name;
        usage += " [options]";

        po::options_description desc_cmdline (usage);
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("run_agas_server,r", "run AGAS server as part of this runtime instance")
            ("console,c", "run this instance as the console")
            ("config", po::value<std::string>(), 
                "load the specified file as an application configuration file")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx,x", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("localities,l", po::value<int>(), 
                "the number of localities to wait for at application startup"
                "(default is 1)")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
            ("pid,p", po::value<int>(), 
                "the number of the node as supplied by the scheduler")
            ("no_hpx_runtime,n", "do not run hpx runtime as part of this runtime instance")
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
        std::cerr << "hpx_runtime: exception caught: " << e.what() << std::endl;
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// helper class for AGAS server initialization
class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
      : agas_pool_(), agas_(agas_pool_, host, port)
    {}

    void run (bool blocking)
    {
        agas_.run(blocking);
    }

private:
    hpx::util::io_service_pool agas_pool_; 
    hpx::naming::resolver_server agas_;
};

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
        std::cerr << "hpx_runtime: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "             using default value instead: " << port << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline("hpx_runtime", argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), agas_host;
        boost::uint16_t hpx_port = HPX_PORT, agas_port = 0;
        int num_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::worker;
        int num_localities = 1;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("console"))
            mode = hpx::runtime::console;
        else if (vm.count("config")) {
            std::cerr << "hpx_runtime: config option ignored, used for console "
                         "instance only\n";
        }

        if (vm.count("localities"))
            num_localities = vm["localities"].as<int>();

        // do we need to execute the HPX runtime
        bool no_hpx_runtime = vm.count("no_hpx_runtime") != 0;

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server")) { 
            // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

            // block if no HPX runtime is needed
            agas_server->run(no_hpx_runtime);
        }

        // execute HPX runtime, if appropriate
        if (!no_hpx_runtime) {
            // initialize and start the HPX runtime
            runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);

            // if we've got a configuration file (as console) we read it in,
            // otherwise this information will be automatically pulled from 
            // the console
            if (mode == hpx::runtime::console && vm.count("config")) {
                std::string config(vm["config"].as<std::string>());
                rt.get_config().load_application_configuration(config.c_str());
            }

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run(num_threads, num_localities);
        }
    }
    catch (hpx::exception const& e) {
        std::cerr << "hpx_runtime: hpx::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (std::exception const& e) {
        std::cerr << "hpx_runtime: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "hpx_runtime: unexpected exception caught\n";
        return -2;
    }
    return 0;
}
