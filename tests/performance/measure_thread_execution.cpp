//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lexical_cast.hpp>
#include <hpx/hpx.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
// this is a empty test thread
threads::thread_state null_thread(threads::thread_self& self, 
    applier::applier& appl)
{
//     naming::id_type gid = 
//         appl.get_thread_manager().get_thread_gid(self.get_thread_id(), appl);
    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state hpx_main(threads::thread_self& self, 
    applier::applier& appl, util::high_resolution_timer& timer,
    std::size_t num_threads)
{
    // schedule a couple of threads
    timer.restart();
    threads::threadmanager& tm = appl.get_thread_manager();
    for (std::size_t i = 0; i < num_threads; ++i) {
        tm.register_work(boost::bind(&null_thread, _1, boost::ref(appl)),
            threads::pending, false);
    }
    double elapsed = timer.elapsed();
    std::cerr << "Elapsed time [s] for thread initialization of " 
              << num_threads << " threads: " << elapsed << " (" 
              << elapsed/num_threads << " per thread)" << std::endl;

    // start measuring
    timer.restart();
    tm.do_some_work();

    // initiate shutdown of the runtime system
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
        std::size_t num_threads = 1;
        std::size_t num_hpx_threads = 1;

        // Check command line arguments.
        if (argc != 7) {
            std::cerr << "Usage: hpx_runtime hpx_addr hpx_port dgas_addr "
                         "dgas_port number_of_threads number_of_threads" 
                      << std::endl;
            std::cerr << "Try: hpx_runtime <your_ip_addr> 7911 "
                         "<your_ip_addr> 7912 1 1000" << std::endl;
            return -3;
        }
        else {
            hpx_host = argv[1];
            hpx_port = boost::lexical_cast<unsigned short>(argv[2]);
            dgas_host = argv[3];
            dgas_port  = boost::lexical_cast<unsigned short>(argv[4]);
            num_threads = boost::lexical_cast<int>(argv[5]);
            num_hpx_threads = boost::lexical_cast<int>(argv[6]);
        }

        // run the DGAS server instance here
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas(dgas_pool, dgas_host, dgas_port, true);

        // initialize and start the HPX runtime
        hpx::runtime rt(dgas_host, dgas_port, hpx_host, hpx_port);

        // the main thread will wait (block) for the shutdown action and 
        // the threadmanager is serving incoming requests in the meantime
        util::high_resolution_timer timer;
        rt.run(boost::bind(hpx_main, _1, _2, boost::ref(timer), num_hpx_threads), 
            num_threads);
        double elapsed = timer.elapsed();
        std::cout << "Elapsed time [s] for " << num_hpx_threads 
                  << " threads: " << elapsed << " (" 
                  << elapsed/num_hpx_threads << " per thread)" << std::endl;
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
