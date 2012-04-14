//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/bind.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
int test0()
{
    return 42;
}
HPX_PLAIN_ACTION(test0, test0_action);

void bind_test0(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_TEST_EQ(hpx::util::bind<test0_action>(id)(), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_1)(id), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_1)(id, 41, 3.0), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_2)(41, id), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test1(int i)
{
    return i;
}
HPX_PLAIN_ACTION(test1, test1_action);

void bind_test1(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, 42)(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, 42)(id), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, _1)(42), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, _2)(id, 42), 42);
}

void function_bind_test1(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_STD_FUNCTION<int()> f1 =
        hpx::util::bind<test1_action>(id, 42);
    HPX_TEST_EQ(f1(), 42);

    HPX_STD_FUNCTION<int(hpx::naming::id_type)> f2 =
        hpx::util::bind<test1_action>(_1, 42);
    HPX_TEST_EQ(f2(id), 42);

    HPX_STD_FUNCTION<int(int)> f3 =
        hpx::util::bind<test1_action>(id, _1);
    HPX_TEST_EQ(f3(42), 42);

    HPX_STD_FUNCTION<int(hpx::naming::id_type, int)> f4 =
        hpx::util::bind<test1_action>(_1, _2);
    HPX_TEST_EQ(f4(id, 42), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test2(hpx::util::function<int(hpx::naming::id_type)> f)
{
    return f(hpx::find_here());
}
HPX_PLAIN_ACTION(test2, test2_action);

void function_bind_test2(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;

    hpx::util::function<int(hpx::naming::id_type)> f1 =
        hpx::util::bind<test1_action>(_1, 42);

    HPX_STD_FUNCTION<int()> f2 =
        hpx::util::bind<test2_action>(id, f1);

    HPX_TEST_EQ(f2(), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test3(hpx::util::function<int()> f)
{
    return f();
}
HPX_PLAIN_ACTION(test3, test3_action);

void function_bind_test3(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;

    hpx::util::function<int()> f1 =
        hpx::util::bind<test1_action>(hpx::find_here(), 42);

    HPX_STD_FUNCTION<int()> f2 =
        hpx::util::bind<test3_action>(id, f1);

    HPX_TEST_EQ(f2(), 42);
}

///////////////////////////////////////////////////////////////////////////////
void run_tests(hpx::naming::id_type id)
{
    bind_test0(id);
    bind_test1(id);
    function_bind_test1(id);
    function_bind_test2(id);
    function_bind_test3(id);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    // run the test on all localities
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    BOOST_FOREACH(hpx::naming::id_type id, localities)
        run_tests(id);

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

