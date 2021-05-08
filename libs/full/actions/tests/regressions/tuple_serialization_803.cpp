//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that issue #803 is resolved (Create proper serialization
// support functions for hpx::tuple).

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/testing.hpp>

#include <string>

typedef hpx::tuple<int, double, std::string> tuple_type;
typedef hpx::tuple<int, double, std::string> tuple_base_type;

void worker1(tuple_type t)
{
    HPX_TEST_EQ(hpx::get<0>(t), 42);
    HPX_TEST_EQ(hpx::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker1);

void worker2(tuple_base_type t)
{
    HPX_TEST_EQ(hpx::get<0>(t), 42);
    HPX_TEST_EQ(hpx::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker2);

void worker1_ref(tuple_type const& t)
{
    HPX_TEST_EQ(hpx::get<0>(t), 42);
    HPX_TEST_EQ(hpx::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker1_ref);

void worker2_ref(tuple_base_type const& t)
{
    HPX_TEST_EQ(hpx::get<0>(t), 42);
    HPX_TEST_EQ(hpx::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker2_ref);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        tuple_type t(42, 3.14, "test");

        worker1_action act1;
        act1(hpx::find_here(), t);

        worker1_ref_action act2;
        act2(hpx::find_here(), t);
    }
    {
        tuple_base_type t(42, 3.14, "test");

        worker2_action act1;
        act1(hpx::find_here(), t);

        worker2_ref_action act2;
        act2(hpx::find_here(), t);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ(hpx::init(argc, argv), 0);

    return hpx::util::report_errors();
}
#endif
