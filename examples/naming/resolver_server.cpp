//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/naming/resolver_server.hpp>

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

int main(int argc, char* argv[])
{
    std::string host;
    unsigned short port;
    std::size_t num_threads;
    
    // Check command line arguments.
    if (argc != 4) 
    {
        std::cerr << "Using default settings: localhost:7911, threads:1" 
                  << std::endl;
        std::cerr << "Possible arguments: <DGAS address> <DGAS port> <num_threads>"
                  << std::endl;

        host = "localhost";
        port = 7911;
        num_threads = 1;
    }
    else
    {
        host = argv[1];
        port = boost::lexical_cast<unsigned short>(argv[2]);
        num_threads = boost::lexical_cast<std::size_t>(argv[3]);
    }

    try {
#if defined(BOOST_WINDOWS)
        // Initialize server.
        hpx::util::io_service_pool io_service_pool(num_threads); 
        hpx::naming::resolver_server s(io_service_pool, host, port, false);

        // Set console control handler to allow server to be stopped.
        console_ctrl_function = boost::bind(&hpx::naming::resolver_server::stop, &s);
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

        std::cout << "Address resolver (server...)" << std::flush << std::endl;
        
        // Run the server until stopped.
        s.run();
#else
        // Block all signals for background thread.
        sigset_t new_mask;
        sigfillset(&new_mask);
        sigset_t old_mask;
        pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask);
        
        // Run server in background thread.
        hpx::util::io_service_pool io_service_pool(num_threads); 
        hpx::naming::resolver_server s(io_service_pool, host, port, false);
        boost::thread t(boost::bind(&hpx::naming::resolver_server::run, &s, true));

        std::cout << "Address resolver (server...)" << std::endl;

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

        // Stop the server.
        s.stop();
        t.join();
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

