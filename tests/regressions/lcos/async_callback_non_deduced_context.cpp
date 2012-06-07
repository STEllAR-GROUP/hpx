//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2012      Bryce Adelstein-Lelbach 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52072

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/bind.hpp>

///////////////////////////////////////////////////////////////////////////////
int test()
{
    return 42;
}
HPX_PLAIN_ACTION(test, test_action);

///////////////////////////////////////////////////////////////////////////////
void future_callback(hpx::lcos::future<int> p)
{
    HPX_TEST(p.has_value());
    HPX_TEST_EQ(p.get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    using hpx::lcos::future;
    using hpx::async_callback;

    {
        test_action do_test;

        future<int> p = async_callback(
            do_test,
            boost::bind(future_callback, _1),
            hpx::find_here()
        );

        HPX_TEST_EQ(p.get(), 42);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}

