//  Copyright (c) 2007-2008 Hartmut Kaiser, Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/components/accumulator/accumulator.hpp>

#include <boost/lexical_cast.hpp>

using namespace hpx;
using namespace std;

///////////////////////////////////////////////////////////////////////////////
threads::thread_state 
hpx_main(threads::thread_self& self, applier::applier& appl)
{
    // create an accumulator locally
    using hpx::components::accumulator;
    accumulator accu (accumulator::create(self, appl, 
        appl.get_runtime_support_gid()));

    // print some message
    std::cout << "accumulator client, you may enter some commands "
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
            std::cout << accu.query(self) << std::endl;
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
        unsigned short hpx_port, dgas_port;

        // Check command line arguments.
        if (argc != 5) {
            std::cerr << "Usage: accumulator_client hpx_addr hpx_port dgas_addr "
                         "dgas_port" << std::endl;
            std::cerr << "Try: accumulator_client <your_ip_addr> 7911 "
                         "<your_ip_addr> 7912" << std::endl;
            return -3;
        }
        else {
            hpx_host = argv[1];
            hpx_port = boost::lexical_cast<unsigned short>(argv[2]);
            dgas_host = argv[3];
            dgas_port  = boost::lexical_cast<unsigned short>(argv[4]);
        }

        // run the DGAS server instance here
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas(dgas_pool, dgas_host, dgas_port, true);

        // initialize and start the HPX runtime
        hpx::runtime rt(dgas_host, dgas_port, hpx_host, hpx_port);
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

//     try {
//         if (argc != 5) {
//             std::cout << "Usage: client <px_ip> <px_port> <gas_ip> <gas_port>" << std::endl;
//             return -1;
//         }
// 
//         // Initialize ParalleX with the ip addresses of the GAS server and the 
//         // accumulator server
//         unsigned short px_port = boost::lexical_cast<unsigned short>(argv[2]);
//         unsigned short gas_port = boost::lexical_cast<unsigned short>(argv[4]);
// 
//         // Start ParalleX services
//         hpx::naming::resolver_client dgas_c(argv[3], gas_port);
//         hpx::parcelset::parcelport ps(dgas_c, argv[1], px_port);
// 
//         px.run(false);
// 
//         std::cout << "px client, you may enter some commands (try 'help' if in doubt...)" << std::endl;
// 
//         // Get a reference to the accumulator that is globally identified as "1"
//         hpx::components::accumulator accu (ps, "/example1/accumulators/1");
// 
//         // execute a couple of commands on this component
//         std::string cmd;
//         std::cin >> cmd;
//         while (true)
//         {
//             if(cmd == "init") {
//                 accu.init();
//             }
//             else if (cmd == "add") {
//                 std::string arg;
//                 std::cin >> arg;
//                 accu.add(boost::lexical_cast<double>(arg));
//             }
//             else if (cmd == "print") {
//                 accu.print();
//             }
//             else if (cmd == "help") {
//                 std::cout << "commands: init, add [amount], print, help, quit" 
//                           << std::endl;
//             }
//             else if (cmd == "quit") {
//                 break;
//             }
//             else {
//                 std::cout << "Invalid command." << std::endl;
//                 std::cout << "commands: init, add [amount], print, help, quit" 
//                           << std::endl;
//             }
//             std::cin >> cmd;
//         }
//         px.stop();
//     }
//     catch (std::exception& e) {
//         std::cerr << "std::exception caught: " << e.what() << std::endl;
//         return -1;
//     }
//     catch (...) {
//         std::cerr << "unexpected exception caught" << std::endl;
//         return -2;
//     }
//     return 0;
// }

