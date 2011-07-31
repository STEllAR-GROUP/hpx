//  Copyright (c) 2007-2011 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Parts of this nqueen_client.cpp has been taken from the accumulator example
//  by Hartmut Kaiser.

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include "nqueen/nqueen.hpp"

using namespace hpx;
using namespace std;

namespace po = boost::program_options;

threads::thread_state_enum hpx_main()
{
    std::vector<naming::id_type> prefixes;
    naming::id_type prefix;
    applier::applier& appl = applier::get_applier();
    if(appl.get_remote_prefixes(prefixes)) {
        prefix = prefixes[0];
    }
    else {
        prefix = appl.get_runtime_support_gid();
    }

    using hpx::components::Board;

    Board board;
    board.create(prefix);

    std::cout << "Enter size of board. Default size is 8." << std::endl;
    std::cout << "Command Options: size[value] | default | print | clear | quit" << std::endl;

    std::string cmd;
    std::cin >> cmd;

    while (std::cin.good())
    {
        if(cmd == "size"){
            std::string arg;
            std::cin >> arg;
            std::size_t sz_temp = boost::lexical_cast<std::size_t>(arg);
            board.initBoard(sz_temp, 0);
            board.solveNqueen(board.accessBoard(), sz_temp, 0);
        }
        else if(cmd == "default"){
            board.initBoard(DS,DS);
            board.solveNqueen(board.accessBoard(), DS, 0);
        }
        else if(cmd == "print"){
            board.printBoard();

        }
        else if(cmd == "clear"){
            board.clearBoard();
        }
        else if (cmd == "quit"){
            break;
        }
        else {
            std::cout << "Invalid Command." <<std::endl;
            std::cout << "Options: size[value] | default | print | clear | quit" << std::endl;
        }
        std::cin >> cmd;
    }
    board.free();

    components::stubs::runtime_support::shutdown_all();

    return threads::terminated;
}

bool parse_commandline(int argc, char *argv[], po::variables_map& vm)
{
    try {
        po::options_description opt_cmdline ("Usage: hpx_runtime [options]");
        opt_cmdline.add_options()
                ("run_agas_server,r", "run AGAS server as part of this runtime instance")
                ("worker,w", "run this instance in worker (non-console) mode")
                            ("agas,a", po::value<std::string>(),
                                "the IP address the AGAS server is running on (default taken "
                                "from hpx.ini), expected format: 192.168.1.1:7912")
                            ("hpx,x", po::value<std::string>(),
                                "the IP address the HPX parcelport is listening on (default "
                                "is localhost:7910), expected format: 192.168.1.1:7913")
                            ("localities,l", po::value<int>(),
                                "the number of localities to wait for at application startup "
                                "(default is 1)")
                            ("threads,t", po::value<int>(),
                                "the number of operating system threads to spawn for this "
                                "HPX locality")
                        ;

        po::store(po::command_line_parser(argc, argv)
            .options(opt_cmdline).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << opt_cmdline;
            return false;
        }
    }
    catch (std::exception const& e) {
        std::cerr << "Exception caught" << e.what() << std::endl;
        return false;
    }
    return true;
}

inline void
    split_ip_address(std::string const& v, std::string& addr, boost::uint16_t& port)
{
    std::string::size_type a = v.find_first_of(":");
    try {
        if (a != std::string::npos) {
            addr = v.substr(0, a);
            port = boost::lexical_cast<boost::uint16_t>(v.substr(a+1));
        }
        else {
            addr = v;
        }
    }
    catch (boost::bad_lexical_cast const& ) {
        std::cerr << "nqueen_client: illegal port number given: " << v.substr(a+1) << std::endl;
        std::cerr << "         using default value instead: " << port << std::endl;
    }
}

//AGAS helper class

class agas_server_helper
{
public:
    agas_server_helper(std::string host, boost::uint16_t port)
        : agas_pool_(), agas_(agas_pool_, host, port)
    {
        agas_.run(false);
    }

private:
    hpx::util::io_service_pool agas_pool_;
    hpx::naming::resolver_server agas_;
};

//runtime type used in the application
typedef hpx::runtime_impl<hpx::threads::policies::global_queue_scheduler> runtime_type;

int main(int argc, char* argv[])
{
    try{
        po::variables_map vm;
        if(!parse_commandline(argc, argv, vm))
            return -1;


        std::string hpx_host(HPX_INITIAL_IP_ADDRESS), agas_host;
        boost::uint16_t hpx_port = HPX_INITIAL_IP_PORT, agas_port = 0;

        int num_threads = 1;
        hpx::runtime_mode mode = hpx::runtime_mode_console;    // default is console mode
        int num_localities = 1;

        // extract IP address/port arguments
        if (vm.count("agas")) 
            split_ip_address(vm["agas"].as<std::string>(), agas_host, agas_port);

        if (vm.count("hpx")) 
            split_ip_address(vm["hpx"].as<std::string>(), hpx_host, hpx_port);

        if (vm.count("threads"))
            num_threads = vm["threads"].as<int>();

        if (vm.count("worker"))
            mode = hpx::runtime_mode_worker;

        if (vm.count("localities"))
            num_localities = vm["localities"].as<int>();

        // initialize and run the AGAS service, if appropriate
        std::auto_ptr<agas_server_helper> agas_server;
        if (vm.count("run_agas_server"))  // run the AGAS server instance here
            agas_server.reset(new agas_server_helper(agas_host, agas_port));

        // initialize and start the HPX runtime
        runtime_type rt(hpx_host, hpx_port, agas_host, agas_port, mode);
        rt.run(hpx_main, num_threads, num_localities);
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
