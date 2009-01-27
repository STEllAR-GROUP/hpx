//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>
#include <boost/detail/lightweight_test.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    HPX_THROW_EXCEPTION(hpx::no_success, "hpx_main", "Some error occurred");
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    try {
        // initialize the AGAS service
        hpx::util::io_service_pool agas_pool; 
        hpx::naming::resolver_server agas(agas_pool);

        // start the HPX runtime
        hpx::runtime rt("localhost", HPX_PORT, "localhost", 0, hpx::runtime::worker);
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
