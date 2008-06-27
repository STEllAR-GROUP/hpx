//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>
#include <hpx/hpx.hpp>


#define MAXITERATIONS 1000
double start_time = 0;

///////////////////////////////////////////////////////////////////////////////
#if defined(BOOST_WINDOWS)

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

#else

#include <pthread.h>
#include <signal.h>

#endif

///////////////////////////////////////////////////////////////////////////////
//  This gets called whenever a parcel was received, it just sends back any 
//  received parcel
void received_parcel(hpx::parcelset::parcelhandler& ph, hpx::naming::address const&)
{
    static int count = 0;
    static double accumulated_time = 0;
    static double turnaround_time =0;
    static std::size_t accumulated_count = 0;
    
    hpx::parcelset::parcel p;
    if (ph.get_parcel(p))
    {
        try {
            accumulated_time += ph.get_current_time() - p.get_start_time();
            ++accumulated_count; 
            turnaround_time +=  ph.get_current_time() - start_time;

            //std::cout << "Received parcel: " << std::hex << p.get_parcel_id() 
            //          << std::flush << std::endl;

            p.set_destination(p.get_source());
            p.set_source(hpx::naming::id_type());
            p.set_parcel_id(hpx::naming::id_type());
            start_time = ph.get_current_time();

            ph.put_parcel(p);
			//std::cout << "Successfully sent parcel: " 
            //        << std::hex << p.get_parcel_id() 
            //        << std::flush << std::endl;

            if (++count >= MAXITERATIONS) {
                std::cout << "Successfully sent " << std::dec << count
                          << " parcels!\nAverage travel time: " 
                          << accumulated_time/accumulated_count
                          << std::flush << std::endl;
                std::cout << "Average turnaround time: " 
                          << turnaround_time/accumulated_count
                          << std::flush << std::endl;
                ph.get_parcelport().stop(false);
                return;
            }
        }
        catch(std::exception const& e) {
            std::cerr << "Caught std::exception: " << e.what() << std::endl;
        }
    }
}

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

            ps_host = "localhost";
            ps_port = 7913;
            gas_host = "localhost";
            gas_port = 7912;
            remote_ps_host = "localhost";
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

#if defined(BOOST_WINDOWS)
        // Start ParalleX services
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_client dgas_c(dgas_pool, gas_host, gas_port);

        hpx::util::io_service_pool io_service_pool(2); 
        hpx::parcelset::parcelport pp (io_service_pool, hpx::naming::locality(ps_host, ps_port));
        hpx::parcelset::parcelhandler ph (dgas_c, pp);

        ph.register_event_handler(received_parcel);

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

        // send initial parcel to remote locality        
        hpx::parcelset::parcel p(remote_prefix);
        start_time = ph.get_current_time();
        hpx::parcelset::parcel_id id = ph.sync_put_parcel(p);

         //std::cout << "Successfully sent parcel: " << std::hex << id 
         //        << std::flush << std::endl;

        pp.run();         // block until stopped
#else
        // Block all signals for background thread.
        sigset_t new_mask;
        sigfillset(&new_mask);
        sigset_t old_mask;
        pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask);
        
        // Start ParalleX services
        hpx::util::io_service_pool io_service_pool(1); 
        hpx::naming::resolver_client dgas_c(io_service_pool, gas_host, gas_port);
        hpx::parcelset::parcelport pp (io_service_pool, hpx::naming::locality(ps_host, ps_port));
        hpx::parcelset::parcelhandler ph (dgas_c, pp);

        ph.register_event_handler(received_parcel);

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
        boost::thread t1(boost::bind(&hpx::parcelset::parcelport::run, &pp, true));

        std::cout << "Parcelset (client) listening at port: " << ps_port 
                  << std::flush << std::endl;

        // send initial parcel to remote locality        
        hpx::parcelset::parcel p(remote_prefix);
        hpx::parcelset::parcel_id id = ph.sync_put_parcel(p);

        std::cout << "Successfully sent parcel: " << std::hex << id 
                  << std::flush << std::endl;

        // Restore previous signals.
        pthread_sigmask(SIG_SETMASK, &old_mask, 0);

        // Wait for signal indicating time to shut down.
        sigset_t wait_mask;
        sigemptyset(&wait_mask);
        sigaddset(&wait_mask, SIGINT);
        sigaddset(&wait_mask, SIGQUIT);
        sigaddset(&wait_mask, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &wait_mask, 0);
        int sig = 0;
        sigwait(&wait_mask, &sig);

        pp.stop();
        t1.join();
#endif
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

