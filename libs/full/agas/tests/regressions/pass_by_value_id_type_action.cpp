//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Probably #431

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test(hpx::id_type) {}
HPX_PLAIN_ACTION(test, test_action)

hpx::id_type test_return()
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(test_return, test_return_action)

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map&)
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    for (hpx::id_type const& id : localities)
    {
        {
            hpx::id_type a = id;

            test_action act;
            hpx::future<void> f = hpx::async(act, id, a);
            f.get();

            HPX_TEST_EQ(id, a);
        }

        {
            hpx::id_type a = id;

            test_action act;
            act(id, a);

            HPX_TEST_EQ(id, a);
        }

        {
            test_return_action act;
            hpx::future<hpx::id_type> f = hpx::async(act, id);

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
    // Initialize and run HPX.
    return hpx::init(argc, argv);
}
#endif
