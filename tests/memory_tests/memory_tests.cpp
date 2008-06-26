//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lexical_cast.hpp>
#include <boost/detail/lightweight_test.hpp>

#include <hpx/hpx.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
threadmanager::thread_state hpx_main(threadmanager::px_thread_self& self, 
    applier::applier& appl)
{
    // try to access some memory directly
    boost::uint32_t value = 0;
    naming::id_type value_gid (naming::get_memory_id(appl.get_prefix(), &value));

    appl.apply<components::server::memory::store32_action>(value_gid, 1);
    BOOST_TEST(value == 1);

    // initiate shutdown of the runtime system
    components::stubs::runtime_support::shutdown_all(appl, 
        naming::get_runtime_support_id(appl.get_prefix()));

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
            std::cerr << "Usage: memory_tests hpx_addr hpx_port dgas_port" 
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
