//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <hpx/hpx.hpp>
#include <hpx/components/accumulator/server/accumulator.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/function.hpp>

//boost::function0<void> console_ctrl_function;
//
//BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
//{
//    switch (ctrl_type) {
//    case CTRL_C_EVENT:
//    case CTRL_BREAK_EVENT:
//    case CTRL_CLOSE_EVENT:
//    case CTRL_SHUTDOWN_EVENT:
//        console_ctrl_function();
//        return TRUE;
//
//    default:
//        return FALSE;
//    }
//}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        std::string ps_host, gas_host;
        boost::uint16_t ps_port, gas_port;
        std::size_t num_threads;

        // Check command line arguments.
        if (argc != 6) 
        {
            std::cerr << "Using default settings: ps:localhost:7911 dgas:localhost:7912 threads:3" 
                << std::endl;
            std::cerr << "Possible arguments: <HPX address> <HPX port> <DGAS address> <DGAS port> <num_threads>"
                << std::endl;

            ps_host = "130.39.128.55";;
            ps_port = 7911;
            gas_host = "130.39.128.55";;
            gas_port = 7912;
            num_threads = 3;
        }
        else
        {
            ps_host = argv[1];
            ps_port = boost::lexical_cast<boost::uint16_t>(argv[2]);
            gas_host = argv[3];
            gas_port  = boost::lexical_cast<boost::uint16_t>(argv[4]);
            num_threads = boost::lexical_cast<std::size_t>(argv[5]);
        }

        // Run the ParalleX services
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas_s(dgas_pool, gas_host, gas_port);
        hpx::naming::resolver_client dgas_c(dgas_pool, 
            hpx::naming::locality(gas_host, gas_port));

        hpx::util::io_service_pool io_service_pool(num_threads); 
        hpx::parcelset::parcelport pp(io_service_pool, 
            hpx::naming::locality(ps_host, ps_port));
        hpx::parcelset::parcelhandler ph(dgas_c, pp);

        // Create a new thread-manager
        hpx::util::io_service_pool timerpool;
        hpx::threads::threadmanager tm(timerpool);
        // Create a new applier
        hpx::applier::applier app(ph, tm, 0, 0);
        // Create a new action-manager
        hpx::actions::action_manager am(app);

        // Set console control handler to allow server to be stopped.
        //console_ctrl_function = 
        //    boost::bind(&hpx::parcelset::parcelport::stop, &pp, true);
        //SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

        // Run the server until stopped.
        std::cout << "Parcelset (server) listening at port: " << ps_port 
            << std::endl;
        std::cout << "GAS server listening at port: " << gas_port 
            << std::flush << std::endl;

///////////////////////////////////////////////////////////////////////////////
// Start test code
///////////////////////////////////////////////////////////////////////////////

        // Create a new accumulator object
        hpx::components::server::accumulator accu;
        // Statically assign a new global-id, should be dynamically done in the future
        hpx::naming::id_type id(99);
        // Put together the host-name and the port-number of the locality
        hpx::naming::locality l(ps_host, ps_port);
        // Get the local virtual address of the accumulator object
//        hpx::naming::address::address_type lva = &accu;
        // Bind the accumulator with the resolver-client
        dgas_c.bind(id, hpx::naming::address(l, 
            hpx::components::server::accumulator::get_component_type(), &accu));

        tm.run();
        pp.run();
        dgas_s.stop();

///////////////////////////////////////////////////////////////////////////////
// End test code
///////////////////////////////////////////////////////////////////////////////
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

