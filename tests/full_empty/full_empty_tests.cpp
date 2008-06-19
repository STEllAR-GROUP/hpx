//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/bind.hpp>
#include <boost/detail/lightweight_test.hpp>

#include <hpx/hpx.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state test1_helper(threadmanager::px_thread_self& self, 
    applier::applier&, hpx::lcos::full_empty<int>& data)
{
    data.set(self, 1);
    BOOST_TEST(!data.is_empty());

    return threadmanager::terminated;
}

threadmanager::thread_state test1(threadmanager::px_thread_self& self, 
    applier::applier& appl)
{
    // create a full_empty data item
    hpx::lcos::full_empty<int> data;
    BOOST_TEST(data.is_empty());

    // schedule the helper thread
    appl.get_thread_manager().register_work(
        boost::bind(&test1_helper, _1, boost::ref(appl), boost::ref(data)));

    // wait for the other thread to set 'data' to full
    int value = 0;
    data.read(self, value);   // this blocks for test1_helper to set value

    BOOST_TEST(!data.is_empty());
    BOOST_TEST(value == 1);

    return threadmanager::terminated;
}


///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state hpx_main(threadmanager::px_thread_self& self, 
    applier::applier& appl)
{
    // schedule test threads: test1
    appl.get_thread_manager().register_work(
        boost::bind(&test1, _1, boost::ref(appl)));

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
            std::cerr << "Usage: full_empty_test hpx_addr hpx_port dgas_port" 
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
        
        // initialize and start the HPX runtime
        hpx::runtime rt(host, dgas_port, host, ps_port);
        rt.run(hpx_main);
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


