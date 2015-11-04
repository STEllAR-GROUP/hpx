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

hpx::naming::id_type test_return() { return hpx::find_here(); }
HPX_PLAIN_ACTION(test_return, test_return_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

    for (hpx::naming::id_type const& id : localities)
    {
        {
            hpx::naming::id_type a = id;

            test_action act;
            hpx::lcos::future<void> f = hpx::async(act, id, a);
            f.get();

            HPX_TEST_EQ(id, a);
        }

        {
            hpx::naming::id_type a = id;

            test_action act;
            act(id, a);

            HPX_TEST_EQ(id, a);
        }

        {

            test_return_action act;
            hpx::lcos::future<hpx::naming::id_type> f = hpx::async(act, id);

            HPX_TEST_EQ(id, f.get());
        }

        {
            test_return_action act;
            HPX_TEST_EQ(act(id), id);
        }
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

