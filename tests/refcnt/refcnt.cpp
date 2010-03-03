//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#include "components/refcnt.hpp"

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description desc_cmdline ("Usage: refcnt [options]");
        desc_cmdline.add_options()
            ("help,h", "print out program usage (this message)")
            ("agas,a", po::value<std::string>(), 
                "the IP address the AGAS server is running on (default taken "
                "from hpx.ini), expected format: 192.168.1.1:7912")
            ("console,c", po::value<std::string>(), 
                "the IP address the HPX parcelport of the console is listening on (default "
                "is localhost:7910), expected format: 192.168.1.1:7913")
            ("worker,w", po::value<std::string>(), 
                "the IP address the HPX parcelport of the worker is listening on (default "
                "is localhost:7911), expected format: 192.168.1.1:7913")
            ("threads,t", po::value<int>(), 
                "the number of operating system threads to spawn for this"
                "HPX locality")
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
        std::cerr << "refcnt: exception caught: " << e.what() << std::endl;
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
        std::cerr << "refcnt: illegal port number given: " << v.substr(p+1) << std::endl;
        std::cerr << "        using default value instead: " << port << std::endl;
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
int hpx_main()
{
    // try to get arguments from application configuration
    runtime& rt = get_runtime();

    // get list of all known localities
    std::vector<naming::gid_type> prefixes;
    applier::applier& appl = applier::get_applier();

    naming::gid_type this_prefix = appl.get_runtime_support_gid();
    naming::gid_type that_prefix;

    if (appl.get_remote_prefixes(prefixes)) {
        // execute the fib() function on any of the remote localities
        that_prefix = prefixes[0];
    }
    else {
        // execute the fib() function locally
        that_prefix = this_prefix;
    }

    // the prefixes don|t need to be managed
    naming::id_type this_(this_prefix, naming::id_type::unmanaged);
    naming::id_type that_(that_prefix, naming::id_type::unmanaged);

    // create instances of the test component and invoke the test action
    components::refcnt_test::refcnt local; 
    local.create(this_);

    local.test();

    // same for remote locality
    components::refcnt_test::refcnt remote; 
    remote.create(that_);

    remote.test();

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// this is the runtime type we use in this application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

///////////////////////////////////////////////////////////////////////////////
// start a runtime instance 
void execute(hpx::runtime::mode mode, int num_threads, 
    std::string host, boost::uint16_t port,
    std::string agas_host, boost::uint16_t agas_port)
{
    runtime_type rt(host, port, agas_host, agas_port, mode);
    if (mode == hpx::runtime::worker) {
        rt.run(num_threads, 2);
    }
    else {
        rt.run(hpx_main, num_threads, 2);
    }
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // analyze the command line
        po::variables_map vm;
        if (!parse_commandline(argc, argv, vm))
            return -1;

        // Check command line arguments.
        std::string console_host("localhost"), worker_host("localhost"), agas_host;
        boost::uint16_t console_port = HPX_PORT, worker_port = HPX_PORT+1, agas_port = 0;
        int num_threads = 1;
        hpx::runtime::mode mode = hpx::runtime::console;    // default is console mode

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("console")) 
            split_ip_address(vm["console"].as<std::string>(), console_host, console_port);

        if (vm.count("worker")) 
            split_ip_address(vm["worker"].as<std::string>(), worker_host, worker_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        // initialize and run the AGAS service
        std::auto_ptr<agas_server_helper> agas_server(
            new agas_server_helper(agas_host, agas_port));

        boost::thread console(execute, hpx::runtime::console, num_threads, 
            console_host, console_port, agas_host, agas_port);
        boost::thread worker(execute, hpx::runtime::worker, num_threads, 
            worker_host, worker_port, agas_host, agas_port);

        worker.join();
        console.join();
    }
    catch (std::exception& e) {
        std::cerr << "refcnt: std::exception caught: " << e.what() << "\n";
        return -1;
    }
    catch (...) {
        std::cerr << "refcnt: unexpected exception caught\n";
        return -2;
    }
    return 0;
}

