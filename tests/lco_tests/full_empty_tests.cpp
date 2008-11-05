//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/bind.hpp>
#include <boost/detail/lightweight_test.hpp>

#include <hpx/hpx.hpp>
#include <hpx/components/accumulator/server/accumulator.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
threads::thread_state test1_helper(threads::thread_self& self, 
    applier::applier& appl, hpx::util::full_empty<int>& data)
{
    // retrieve gid for this thread
    naming::id_type gid = 
        appl.get_thread_manager().get_thread_gid(self.get_thread_id(), appl);
    BOOST_TEST(gid);

    data.set(1);
    BOOST_TEST(!data.is_empty());

    return threads::terminated;
}

threads::thread_state test1(threads::thread_self& self, 
    applier::applier& appl)
{
    // retrieve gid for this thread
    naming::id_type gid = 
        appl.get_thread_manager().get_thread_gid(self.get_thread_id(), appl);
    BOOST_TEST(gid);

    // create a full_empty data item
    hpx::util::full_empty<int> data;
    BOOST_TEST(data.is_empty());

    // schedule the helper thread
    register_work(appl, 
        boost::bind(&test1_helper, _1, boost::ref(appl), boost::ref(data)));

    // wait for the other thread to set 'data' to full
    int value = 0;
    data.read(self, value);   // this blocks for test1_helper to set value

    BOOST_TEST(!data.is_empty());
    BOOST_TEST(value == 1);

    value = 0;
    data.read(self, value);   // this should not block anymore

    BOOST_TEST(!data.is_empty());
    BOOST_TEST(value == 1);

    return threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
threads::thread_state hpx_main(threads::thread_self& self, 
    applier::applier& appl)
{
    // retrieve gid for this thread
    naming::id_type gid = 
        appl.get_thread_manager().get_thread_gid(self.get_thread_id(), appl);
    BOOST_TEST(gid);

    // schedule test threads: test1
    register_work(appl, test1);

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
        boost::uint16_t ps_port, agas_port;

        // Check command line arguments.
        if (argc != 4) {
            std::cerr << "Usage: full_empty_test hpx_addr hpx_port agas_port" 
                << std::endl;
            return -1;
        }
        else {
            host = argv[1];
            ps_port = boost::lexical_cast<boost::uint16_t>(argv[2]);
            agas_port  = boost::lexical_cast<boost::uint16_t>(argv[3]);
        }

        // initialize the AGAS service
        hpx::util::io_service_pool agas_pool; 
        hpx::naming::resolver_server agas(agas_pool, host, agas_port);

        // initialize and start the HPX runtime
        hpx::runtime rt(host, ps_port, host, agas_port);
        rt.run(hpx_main, 2);
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


