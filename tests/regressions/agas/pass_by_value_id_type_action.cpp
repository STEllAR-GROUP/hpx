//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Probably #431

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
void test(hpx::naming::id_type id) {}
HPX_PLAIN_ACTION(test, test_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    {
        hpx::naming::id_type a = hpx::find_here();

        test_action act;
        hpx::lcos::future<void> f = hpx::async(act, hpx::find_here(), a);
        f.get();

        HPX_TEST_EQ(hpx::find_here(), a);
    }

    {
        hpx::naming::id_type a = hpx::find_here();

        test_action act;
        act(hpx::find_here(), a);

        HPX_TEST_EQ(hpx::find_here(), a);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

