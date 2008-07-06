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
threadmanager::thread_state timed_set_state_test(
    threadmanager::px_thread_self& self, applier::applier& appl,
    util::high_resolution_timer& timer, double wait_time)
{
    double elapsed = timer.elapsed();
    BOOST_TEST(elapsed >= wait_time);
    std::cerr << "Elapsed: " << elapsed << std::endl;
    return threadmanager::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state duration_set_state_test(
    threadmanager::px_thread_self& self, applier::applier& appl)
{
    util::high_resolution_timer timer;

    threadmanager::thread_id_type id1 = appl.get_thread_manager().register_work(
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 1.0), 
        threadmanager::suspended);

    threadmanager::thread_id_type id2 = appl.get_thread_manager().register_work(
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 2.0), 
        threadmanager::suspended);

    appl.get_thread_manager().timed_set_state(
        boost::posix_time::seconds(1), id1, threadmanager::pending);
    appl.get_thread_manager().timed_set_state(
        boost::posix_time::seconds(2), id2, threadmanager::pending);

    return threadmanager::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state time_set_state_test(
    threadmanager::px_thread_self& self, applier::applier& appl)
{
    util::high_resolution_timer timer;

    threadmanager::thread_id_type id1 = appl.get_thread_manager().register_work(
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 1.0), 
        threadmanager::suspended);

    threadmanager::thread_id_type id2 = appl.get_thread_manager().register_work(
        boost::bind(timed_set_state_test, _1, boost::ref(appl), timer, 2.0), 
        threadmanager::suspended);

    boost::posix_time::ptime now (
        boost::posix_time::microsec_clock::universal_time());

    appl.get_thread_manager().timed_set_state(
        now + boost::posix_time::seconds(1), id1, threadmanager::pending);
    appl.get_thread_manager().timed_set_state(
        now + boost::posix_time::seconds(2), id2, threadmanager::pending);

    return threadmanager::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state hpx_main(threadmanager::px_thread_self& self, 
    applier::applier& appl)
{
    // test timed_set_state using a time duration
    appl.get_thread_manager().register_work(
        boost::bind(duration_set_state_test, _1, boost::ref(appl)));

    // test timed_set_state using a fixed time 
    appl.get_thread_manager().register_work(
        boost::bind(time_set_state_test, _1, boost::ref(appl)));

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all(appl, 
        appl.get_runtime_support_gid());

    return threadmanager::terminated;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // Check command line arguments.
        std::string host;
        unsigned short ps_port, dgas_port;

        // Check command line arguments.
        if (argc != 4) {
            std::cerr << "Usage: timed_set_state_tests hpx_addr hpx_port dgas_port" 
                << std::endl;
            return -1;
        }
        else {
            host = argv[1];
            ps_port = boost::lexical_cast<unsigned short>(argv[2]);
            dgas_port  = boost::lexical_cast<unsigned short>(argv[3]);
        }

        // initialize the DGAS service
        hpx::util::io_service_pool dgas_pool; 
        hpx::naming::resolver_server dgas(dgas_pool, host, dgas_port, true);

        // start the HPX runtime using different numbers of threads
        for (int i = 1; i <= 8; ++i) {
            hpx::runtime rt(host, dgas_port, host, ps_port);
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
