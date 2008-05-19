//  Copyright (c) 2007-2008 Hartmut Kaiser, Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <hpx/hpx.hpp>

using namespace hpx;
using namespace std;
using hpx::components::accumulator;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        if (argc != 5) {
            std::cout << "Usage: client <px_ip> <px_port> <gas_ip> <gas_port>" << std::endl;
            return -1;
        }

        // Initialize ParalleX with the ip addresses of the GAS server and the 
        // accumulator server
        unsigned short px_port = boost::lexical_cast<unsigned short>(argv[2]);
        unsigned short gas_port = boost::lexical_cast<unsigned short>(argv[4]);

        // Start ParalleX services
        hpx::naming::resolver_client dgas_c(argv[3], gas_port);
        hpx::parcelset::parcelport ps(dgas_c, argv[1], px_port);

        px.run(false);

        std::cout << "px client, you may enter some commands (try 'help' if in doubt...)" << std::endl;

        // Get a reference to the accumulator that is globally identified as "1"
        hpx::components::accumulator accu (ps, "/example1/accumulators/1");

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
            else if (cmd == "help") {
                std::cout << "commands: init, add [amount], print, help, quit" 
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
        px.stop();
    }
    catch (std::exception& e) {
        std::cerr << "std::exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "unexpected exception caught" << std::endl;
        return -2;
    }
    return 0;
}

