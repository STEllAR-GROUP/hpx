//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/barrier.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/detail/atomic_count.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
threads::thread_state timed_set_state_test(
    threads::thread_self& self, applier::applier& appl,
    util::high_resolution_timer& timer, double wait_time)
{
    double elapsed = timer.elapsed();
    BOOST_TEST(elapsed + 0.01 >= wait_time);    // we need some leeway here...
    std::cerr << "Elapsed: " << elapsed << std::endl;
    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state duration_set_state_test(
    threads::thread_self& self, applier::applier& appl)
{
    util::high_resolution_timer timer;

    threads::thread_id_type id1 = register_work(appl, 
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 1.0), 
        threads::suspended);

    threads::thread_id_type id2 = register_work(appl, 
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 2.0), 
        threads::suspended);

    set_thread_state(id1, threads::pending, boost::posix_time::seconds(1));
    set_thread_state(id2, threads::pending, boost::posix_time::seconds(2));

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state time_set_state_test(
    threads::thread_self& self, applier::applier& appl)
{
    util::high_resolution_timer timer;

    threads::thread_id_type id1 = register_work(appl, 
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 1.0), 
        threads::suspended);

    threads::thread_id_type id2 = register_work(appl, 
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 2.0), 
        threads::suspended);

    boost::posix_time::ptime now (
        boost::posix_time::microsec_clock::universal_time());

    set_thread_state(id1, threads::pending, now + boost::posix_time::seconds(1));
    set_thread_state(id2, threads::pending, now + boost::posix_time::seconds(2));

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state hpx_main(threads::thread_self& self, 
    applier::applier& appl)
{
    // test timed set_state using a time duration
    register_work(appl, duration_set_state_test);

    // test timed set_state using a fixed time 
    register_work(appl, time_set_state_test);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all(appl);

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // Check command line arguments.
        std::string host;
        boost::uint16_t ps_port, dgas_port;

        // Check command line arguments.
        if (argc != 4) {
            std::cerr << "Usage: timed_set_state_tests hpx_addr hpx_port dgas_port" 
                << std::endl;
            return -1;
        }
        else {
            host = argv[1];
            ps_port = boost::lexical_cast<boost::uint16_t>(argv[2]);
            dgas_port  = boost::lexical_cast<boost::uint16_t>(argv[3]);
        }

        // initialize the DGAS service
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas(dgas_pool, 
            hpx::naming::locality(host, dgas_port));

        // start the HPX runtime using different numbers of threads
        for (int i = 1; i <= 8; ++i) {
            hpx::runtime rt(host, ps_port, host, dgas_port);
            rt.run(hpx_main, i);
        }
    }
    catch (std::exception& e) {
        BOOST_TEST(false);
        std::cerr << "std::exception caught: " << e.what() << "\n";
    }
    catch (...) {
        BOOST_TEST(false);
        std::cerr << "unexpected exception caught\n";
    }
    return boost::report_errors();
}
