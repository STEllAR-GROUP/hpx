//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that issue #803 is resolved (Create proper serialization
// support functions for util::tuple).

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

typedef hpx::util::tuple<int, double, std::string> tuple_type;
typedef hpx::util::tuple<int, double, std::string> tuple_base_type;

void worker1(tuple_type t)
{
    HPX_TEST_EQ(hpx::util::get<0>(t), 42);
    HPX_TEST_EQ(hpx::util::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::util::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker1);

void worker2(tuple_base_type t)
{
    HPX_TEST_EQ(hpx::util::get<0>(t), 42);
    HPX_TEST_EQ(hpx::util::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::util::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker2);

void worker1_ref(tuple_type const& t)
{
    HPX_TEST_EQ(hpx::util::get<0>(t), 42);
    HPX_TEST_EQ(hpx::util::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::util::get<2>(t), "test");
}
HPX_PLAIN_ACTION(worker1_ref);

void worker2_ref(tuple_base_type const&t)
{
    HPX_TEST_EQ(hpx::util::get<0>(t), 42);
    HPX_TEST_EQ(hpx::util::get<1>(t), 3.14);
    HPX_TEST_EQ(hpx::util::get<2>(t), "test");
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

