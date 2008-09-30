//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lexical_cast.hpp>
#include <hpx/hpx.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // Check command line arguments.
        std::string hpx_host, dgas_host;
        unsigned short hpx_port, dgas_port;
        bool is_dgas_server = false;

        // Check command line arguments.
        if (argc != 6) {
            std::cerr << "Usage: hpx_runtime hpx_addr hpx_port dgas_addr "
                         "dgas_port is_dgas_server" << std::endl;
            std::cerr << "Try: hpx_runtime <your_ip_addr> 7911 "
                         "<your_ip_addr> 7912 1" << std::endl;
            return -3;
        }
        else {
            hpx_host = argv[1];
            hpx_port = boost::lexical_cast<unsigned short>(argv[2]);
            dgas_host = argv[3];
            dgas_port  = boost::lexical_cast<unsigned short>(argv[4]);
            is_dgas_server = boost::lexical_cast<int>(argv[5]) ? true : false;
        }

        // initialize the DGAS service
        if (is_dgas_server) {
            // run the DGAS server instance here
            hpx::util::io_service_pool dgas_pool; 
            hpx::naming::resolver_server dgas(dgas_pool, dgas_host, dgas_port, true);

            // initialize and start the HPX runtime
            hpx::runtime rt(dgas_host, dgas_port, hpx_host, hpx_port);

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run();
        }
        else {
            // initialize and start the HPX runtime
            hpx::runtime rt(dgas_host, dgas_port, hpx_host, hpx_port);

            // the main thread will wait (block) for the shutdown action and 
            // the threadmanager is serving incoming requests in the meantime
            rt.run();
        }
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
