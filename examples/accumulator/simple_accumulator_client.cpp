//  Copyright (c) 2007-2008 Hartmut Kaiser, Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/components/simple_accumulator/simple_accumulator.hpp>

#include <boost/lexical_cast.hpp>

using namespace hpx;
using namespace std;

///////////////////////////////////////////////////////////////////////////////
threads::thread_state 
hpx_main(threads::thread_self& self, applier::applier& appl)
{
    // get list of all known localities
    std::vector<naming::id_type> prefixes;
    naming::id_type prefix;
    if (appl.get_remote_prefixes(prefixes)) {
        // create accumulator on any of the remote localities
        prefix = prefixes[0];
    }
    else {
        // create an accumulator locally
        prefix = appl.get_runtime_support_gid();
    }

    using hpx::components::simple_accumulator;
    simple_accumulator accu(simple_accumulator::create(self, appl, prefix));

    // print some message
    std::cout << "simple accumulator client, you may enter some commands "
                 "(try 'help' if in doubt...)" << std::endl;

    // execute a couple of commands on this component
    std::string cmd;
    std::cin >> cmd;
    while (true)
    {
        if(cmd == "init") {
            accu.init();
        }
        else if (cmd == "add") {
            std::string arg;
            std::cin >> arg;
            accu.add(boost::lexical_cast<double>(arg));
        }
        else if (cmd == "print") {
            accu.print();
        }
        else if (cmd == "query") {
            double d = accu.query(self);
            std::cout << d << std::endl;
        }
        else if (cmd == "help") {
            std::cout << "commands: init, add [amount], print, query, help, quit" 
                      << std::endl;
        }
        else if (cmd == "quit") {
            break;
        }
        else {
            std::cout << "Invalid command." << std::endl;
            std::cout << "commands: init, add [amount], print, help, quit" 
                      << std::endl;
        }
        std::cin >> cmd;
    }

    // free the accumulator component
    accu.free();     // this invalidates the remote reference

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all(appl);

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // Check command line arguments.
        std::string hpx_host, dgas_host;
        boost::uint16_t hpx_port, dgas_port;

        // Check command line arguments.
        if (argc != 5) {
            std::cerr << "Usage: simple_accumulator_client hpx_addr hpx_port dgas_addr "
                         "dgas_port" << std::endl;
            std::cerr << "Try: simple_accumulator_client <your_ip_addr> 7911 "
                         "<your_ip_addr> 7912" << std::endl;
            return -3;
        }
        else {
            hpx_host = argv[1];
            hpx_port = boost::lexical_cast<boost::uint16_t>(argv[2]);
            dgas_host = argv[3];
            dgas_port  = boost::lexical_cast<boost::uint16_t>(argv[4]);
        }

        // run the DGAS server instance here
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas(dgas_pool, 
            hpx::naming::locality(dgas_host, dgas_port));

        // initialize and start the HPX runtime
        hpx::runtime rt(hpx_host, hpx_port, dgas_host, dgas_port);
        rt.run(hpx_main);

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

