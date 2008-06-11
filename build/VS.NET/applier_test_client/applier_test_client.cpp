//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <hpx/hpx.hpp>
#include <hpx/components/server/accumulator.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/function.hpp>

boost::function0<void> console_ctrl_function;

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
    switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        console_ctrl_function();
        return TRUE;

    default:
        return FALSE;
    }
}

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
        console_ctrl_function = 
            boost::bind(&hpx::parcelset::parcelport::stop, &pp, true);
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

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
        // 1. Create a static accumulator (gid = 11) on the server side
        // 2. Create a static accumulator (gid = 44) on the client side
        // 3. Send a parcel from client to server, destined at the accumulator
        // 4. Server side receives the parcel
        // 5. Parcel-set will decode the parcel and call AM's call-back fx
        // 6. AM calls the appropriate execute function on the destination obj
        // 7. TM executes the function
        // 8. The function will call the applier to send a parcel back to the client's accumulator
        // 9. The accumulator will add this value
        // 10. Client calls the client's accumulator to print out the value

        // Create a accumulator with static gid of 44 on the client side
        hpx::components::server::accumulator accu;
        hpx::naming::id_type local_id(44);
        hpx::naming::locality l(ps_host, ps_port);
        dgas_c.bind(local_id, hpx::naming::address(l, hpx::components::server::accumulator::value, &accu));

        // Test code to verify that the applier can successfully apply to local components
        hpx::parcelset::parcel_id p_l1 = app.apply<hpx::components::server::accumulator::init_action>(local_id);
        hpx::parcelset::parcel_id p_l2 = app.apply<hpx::components::server::accumulator::add_action>(local_id, 12);
        hpx::parcelset::parcel_id p_l3 = app.apply<hpx::components::server::accumulator::print_action>(local_id);

        // Create a static gid for remote accumulator on server
        hpx::naming::id_type remote_id(11);

        // Test code to verify that the applier can successfully apply to remote components
        hpx::parcelset::parcel_id p_r1 = app.apply<hpx::components::server::accumulator::init_action>(remote_id);
        hpx::parcelset::parcel_id p_r2 = app.apply<hpx::components::server::accumulator::add_action>(remote_id, 13);
        hpx::parcelset::parcel_id p_r3 = app.apply<hpx::components::server::accumulator::print_action>(remote_id);

        //// Send a parcel from client to server, destined at the accumulator (gid = 11)
        //hpx::naming::id_type remote_id(11);
        //// Need to update this to the parcel-return action
        //hpx::parcelset::parcel p_init(remote_id, new hpx::components::server::accumulator::init_action());
        //hpx::parcelset::parcel_id p_id = ph.sync_put_parcel(p_init);

        std::cout << "End of test code" << std::flush << std::endl;

        tm.run();
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

