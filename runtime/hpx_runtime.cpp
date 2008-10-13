//  Copyright (c) 2007-2008 Hartmut Kaiser
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
            ("run_dgas_server,r", "run DGAS server as part of this runtime instance")
            ("no_hpx_runtime,n", "do not run hpx runtime as part of this runtime instance")
            ("dgas", po::value<std::string>(), 
                "the IP address the DGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("hpx", po::value<std::string>(), 
                "the IP address the HPX parcelport is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
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
// helper class for DGAS server initialization
class dgas_server_helper
{
public:
    dgas_server_helper(std::string host, boost::uint16_t port)
      : dgas_pool_(), dgas_(dgas_pool_, host, port)
    {}

    void run (bool blocking)
    {
        dgas_.run(blocking);
    }

private:
    hpx::util::io_service_pool dgas_pool_; 
    hpx::naming::resolver_server dgas_;
};

///////////////////////////////////////////////////////////////////////////////
inline void 
split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    try {
        std::string::size_type p = v.find_first_of(":");
        if (p != std::string::npos) {
            addr = v.substr(0, p);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& /*e*/) {
        ;   // ignore bad_cast exceptions
    }
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline("hpx_runtime", argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string hpx_host("localhost"), dgas_host;
        boost::uint16_t hpx_port = HPX_PORT, dgas_port = 0;

        // extract IP address/port arguments
        if (vm.count("dgas")) 
            split_ip_address(vm["dgas"].as<std::string>(), dgas_host, dgas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        // do we need to execute the HPX runtime
        bool no_hpx_runtime = vm.count("no_hpx_runtime") != 0;

        // initialize and run the DGAS service, if appropriate
        std::auto_ptr<dgas_server_helper> dgas_server;
        if (vm.count("run_dgas_server")) { 
            // run the DGAS server instance here
            dgas_server.reset(new dgas_server_helper(dgas_host, dgas_port));

            // block if no HPX runtime is needed
            dgas_server->run(no_hpx_runtime);
        }

        // execute HPX runtime, if appropriate
        if (!no_hpx_runtime) {
            // initialize and start the HPX runtime
            hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port);

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run();
        }
    }
    catch (hpx::exception const& e) {
        std::cerr << "hpx::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (std::exception const& e) {
        std::cerr << "std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught\n";
        return -2;
    }
    return 0;
}
