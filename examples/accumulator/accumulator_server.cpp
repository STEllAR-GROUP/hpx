//  Copyright (c) 2007-2008 Hartmut Kaiser, Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/lexical_cast.hpp>
#include <hpx/hpx.hpp>

using namespace hpx;

int main(int argc, char* argv[])
{
    try {
        if (argc != 5) {
            std::cout << "Usage: server <px_ip> <px_port> <gas_ip> <gas_port>" << std::endl;
            return -1;
        }

        // Initialize ParalleX with GAS server's ip address
        unsigned short px_port = boost::lexical_cast<unsigned short>(argv[2]);
        unsigned short gas_port = boost::lexical_cast<unsigned short>(argv[4]);

        // Run the ParalleX services
        hpx::naming::resolver_server dgas_s(argv[3], gas_port, true);
        hpx::naming::resolver_client dgas_c(argv[3], gas_port);
        hpx::parcelset::parcelport ps(dgas_c, argv[1], px_port);

        // Create an accumulator with local id (name) 1, global id will be 
        // assigned automatically
        std::auto_ptr<components::accumulator> accumulator(
            px.create<components::accumulator>("/example1/accumulators/1"));

        // Run the ParalleX services
        ps.run(false);

        char line[64];
        std::cout << "Press enter to shutdown the px server..." << std::endl;
        std::cin.getline(line, 64);

        ps.stop();
        dgas_s.stop();
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

