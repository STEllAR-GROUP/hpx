//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <hpx/hpx.hpp>

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
//  This gets called whenever a parcel was received, it just sends back any 
//  received parcel

int main(int argc, char* argv[])
{
    try {
        // Check command line arguments.
        std::string ps_host, remote_ps_host, gas_host;
        unsigned short ps_port, remote_ps_port, gas_port;

        // Check command line arguments.
        if (argc != 7) 
        {
            std::cerr << "Using default settings: ps:localhost:7913 dgas:localhost:7912 remoteps:localhost:7911" 
                << std::endl;
            std::cerr << "Possible arguments: <HPX address> <HPX port> <DGAS address> <DGAS port> <remote HPX address> <remote HPX port>"
                << std::endl;

            ps_host = "130.39.128.55";
            ps_port = 7913;
            gas_host = "130.39.128.55";;
            gas_port = 7912;
            remote_ps_host = "130.39.128.55";;
            remote_ps_port = 7911;
        }
        else
        {
            ps_host = argv[1];
            ps_port = boost::lexical_cast<unsigned short>(argv[2]);
            gas_host = argv[3];
            gas_port  = boost::lexical_cast<unsigned short>(argv[4]);
            remote_ps_host = argv[5];
            remote_ps_port = boost::lexical_cast<unsigned short>(argv[6]);
        }

        // Start ParalleX services
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_client dgas_c(dgas_pool, gas_host, gas_port);

        hpx::util::io_service_pool io_service_pool(2); 
        hpx::parcelset::parcelport pp (io_service_pool, hpx::naming::locality(ps_host, ps_port));
        hpx::parcelset::parcelhandler ph (dgas_c, pp);

        // Create a new thread-manager
        hpx::threadmanager::threadmanager tm;
        // Create a new applier
        hpx::applier::applier app(dgas_c, ph, tm);
        // Create a new action-manager
        hpx::action_manager::action_manager am(app);

        // Set console control handler to allow client to be stopped.
//        console_ctrl_function = 
//            boost::bind(&hpx::parcelset::parcelport::stop, &pp, true);
//        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

        // sleep for a second to give parcelset server a chance to startup
        boost::xtime xt;
        boost::xtime_get(&xt, boost::TIME_UTC);
        xt.sec += 1;
        boost::thread::sleep(xt);

        // retrieve prefix for remote locality
        hpx::naming::id_type remote_prefix;
        hpx::naming::locality remote_l(remote_ps_host, remote_ps_port);
        dgas_c.get_prefix(remote_l, remote_prefix);

        // start parcelport receiver thread
        pp.run(false);

        std::cout << "Parcelset (client) listening at port: " << ps_port 
            << std::flush << std::endl;

///////////////////////////////////////////////////////////////////////////////
// Start test code
///////////////////////////////////////////////////////////////////////////////

        // Test plan
        // 1. Send a parcel from client to server, destined at the accumulator
        // 2. Server side receives the parcel
        // 3. Parcel-set will decode the parcel and call AM's call-back fx
        // 4. AM calls the appropriate execute function on the destination obj
        // 5. TM executes the function

        // statically defined component id is 99
        hpx::naming::id_type id(99);
        
        // create an initial accumulator parcel to send to remote locality
        hpx::parcelset::parcel p_init(id, new hpx::components::server::detail::accumulator::init_action());
        hpx::parcelset::parcel_id p_id = ph.sync_put_parcel(p_init);

        // add 5 to the accumulator
        hpx::parcelset::parcel p_add(id, new hpx::components::server::detail::accumulator::add_action(15));
        hpx::parcelset::parcel_id p_id2 = ph.sync_put_parcel(p_add);

        // print the value of the accumulator
        hpx::parcelset::parcel p_print(id, new hpx::components::server::detail::accumulator::print_action());
        hpx::parcelset::parcel_id p_id3 = ph.sync_put_parcel(p_print);

        std::cout << "Successfully sent parcel: " << std::hex << id 
                  << std::flush << std::endl;

        pp.run();         // block until stopped

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

