//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/bind.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

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

void bind_test1(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    test0_action do_test0;

    HPX_TEST_EQ(hpx::util::bind(do_test0, id)(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _1)(id), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _1)(id, 41, 3.0), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _2)(41, id), 42);
}

void async_test0(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_TEST_EQ(hpx::async(&test0).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind(&test0)).get(), 42);

    HPX_TEST_EQ(hpx::util::bind<test0_action>(id).async().get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_1).async(id).get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_1).async(id, 41, 3.0).get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test0_action>(_2).async(41, id).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind<test0_action>(id)).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test0_action>(_1), id).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test0_action>(_1), id, 41, 3.0).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test0_action>(_2), 41, id).get(), 42);

    HPX_TEST_EQ(hpx::async<test0_action>(id).get(), 42);
}

void async_test1(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    test0_action do_test0;

    HPX_TEST_EQ(hpx::util::bind(do_test0, id).async().get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _1).async(id).get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _1).async(id, 41, 3.0).get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test0, _2).async(41, id).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test0, id)).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test0, _1), id).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test0, _1), id, 41, 3.0).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test0, _2), 41, id).get(), 42);

    HPX_TEST_EQ(hpx::async(do_test0, id).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test1(int i)
{
    return i;
}
HPX_PLAIN_ACTION(test1, test1_action);

void bind_test2(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, 42)(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, 42)(id), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, _1)(42), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, _2)(id, 42), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_2, _1)(42, id), 42);
}

void bind_test3(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    test1_action do_test1;

    HPX_TEST_EQ(hpx::util::bind(do_test1, id, 42)(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _1, 42)(id), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, id, _1)(42), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _1, _2)(id, 42), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _2, _1)(42, id), 42);
}

void async_test2(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    HPX_TEST_EQ(hpx::async(&test1, 42).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind(&test1, 42)).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(&test1, _1), 42).get(), 42);

    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, 42).async().get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, 42).async(id).get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(id, _1).async(42).get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_1, _2).async(id, 42).get(), 42);
    HPX_TEST_EQ(hpx::util::bind<test1_action>(_2, _1).async(42, id).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind<test1_action>(id, 42)).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test1_action>(_1, 42), id).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test1_action>(id, _1), 42).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test1_action>(_1, _2), id, 42).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind<test1_action>(_2, _1), 42, id).get(), 42);

    HPX_TEST_EQ(hpx::async<test1_action>(id, 42).get(), 42);
}

void async_test3(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    test1_action do_test1;

    HPX_TEST_EQ(hpx::util::bind(do_test1, id, 42).async().get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _1, 42).async(id).get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, id, _1).async(42).get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _1, _2).async(id, 42).get(), 42);
    HPX_TEST_EQ(hpx::util::bind(do_test1, _2, _1).async(42, id).get(), 42);

    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test1, id, 42)).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test1, _1, 42), id).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test1, id, _1), 42).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test1, _1, _2), id, 42).get(), 42);
    HPX_TEST_EQ(hpx::async(hpx::util::bind(do_test1, _2, _1), 42, id).get(), 42);

    HPX_TEST_EQ(hpx::async(do_test1, id, 42).get(), 42);
}

void function_bind_test1(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::util::function_nonser<int()> f1 =
        hpx::util::bind<test1_action>(id, 42);
    HPX_TEST_EQ(f1(), 42);

    hpx::util::function_nonser<int(hpx::naming::id_type)> f2 =
        hpx::util::bind<test1_action>(_1, 42);
    HPX_TEST_EQ(f2(id), 42);

    hpx::util::function_nonser<int(int)> f3 =
        hpx::util::bind<test1_action>(id, _1);
    HPX_TEST_EQ(f3(42), 42);

    hpx::util::function_nonser<int(hpx::naming::id_type, int)> f4 =
        hpx::util::bind<test1_action>(_1, _2);
    HPX_TEST_EQ(f4(id, 42), 42);
}

void function_bind_test2(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    test1_action do_test1;

    hpx::util::function_nonser<int()> f1 =
        hpx::util::bind(do_test1, id, 42);
    HPX_TEST_EQ(f1(), 42);

    hpx::util::function_nonser<int(hpx::naming::id_type)> f2 =
        hpx::util::bind(do_test1, _1, 42);
    HPX_TEST_EQ(f2(id), 42);

    hpx::util::function_nonser<int(int)> f3 =
        hpx::util::bind(do_test1, id, _1);
    HPX_TEST_EQ(f3(42), 42);

    hpx::util::function_nonser<int(hpx::naming::id_type, int)> f4 =
        hpx::util::bind(do_test1, _1, _2);
    HPX_TEST_EQ(f4(id, 42), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test2(hpx::util::function<int(hpx::naming::id_type)> f)
{
    return f(hpx::find_here());
}
HPX_PLAIN_ACTION(test2, test2_action);

void function_bind_test3(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;
    test1_action do_test1;

    hpx::util::function<int(hpx::naming::id_type)> f1 =
        hpx::util::bind(do_test1, _1, 42);

    test2_action do_test2;
    hpx::util::function_nonser<int()> f2 =
        hpx::util::bind(do_test2, id, f1);

    HPX_TEST_EQ(f2(), 42);
}

void function_bind_test4(hpx::naming::id_type id)
{
    using hpx::util::placeholders::_1;

    hpx::util::function<int(hpx::naming::id_type)> f1 =
        hpx::util::bind<test1_action>(_1, 42);

    hpx::util::function_nonser<int()> f2 =
        hpx::util::bind<test2_action>(id, f1);

    HPX_TEST_EQ(f2(), 42);
}

///////////////////////////////////////////////////////////////////////////////
int test3(hpx::util::function<int()> f)
{
    return f();
}
HPX_PLAIN_ACTION(test3, test3_action);

void function_bind_test5(hpx::naming::id_type id)
{
    hpx::util::function<int()> f1 =
        hpx::util::bind<test1_action>(hpx::find_here(), 42);

    hpx::util::function_nonser<int()> f2 =
        hpx::util::bind<test3_action>(id, f1);

    HPX_TEST_EQ(f2(), 42);
}

void function_bind_test6(hpx::naming::id_type id)
{
    test1_action do_test1;
    hpx::util::function<int()> f1 =
        hpx::util::bind(do_test1, hpx::find_here(), 42);

    test3_action do_test3;
    hpx::util::function_nonser<int()> f2 =
        hpx::util::bind(do_test3, id, f1);

    HPX_TEST_EQ(f2(), 42);
}

///////////////////////////////////////////////////////////////////////////////
struct A
{
    typedef int result_type;

    int test0()
    {
        return 42;
    }

    int test1(int i)
    {
        return i;
    }
};

void member_bind_test0()
{
    using hpx::util::placeholders::_1;

    A a;

    HPX_TEST_EQ(hpx::util::bind(&A::test0, &a)(), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test0, _1)(&a), 42);

    HPX_TEST_EQ(hpx::util::bind(&A::test0, a)(), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test0, _1)(a), 42);
}

void member_bind_test1()
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    A a;

    HPX_TEST_EQ(hpx::util::bind(&A::test1, &a, 42)(), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, &a, _1)(42), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _1, 42)(&a), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _1, _2)(&a, 42), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _2, _1)(42, &a), 42);

    HPX_TEST_EQ(hpx::util::bind(&A::test1, a, _1)(42), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _1, 42)(a), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _1, _2)(a, 42), 42);
    HPX_TEST_EQ(hpx::util::bind(&A::test1, _2, _1)(42, a), 42);
}

///////////////////////////////////////////////////////////////////////////////
void run_tests(hpx::naming::id_type id)
{
    bind_test0(id);
    bind_test1(id);
    bind_test2(id);
    bind_test3(id);

    function_bind_test1(id);
    function_bind_test2(id);
    function_bind_test3(id);
    function_bind_test4(id);
    function_bind_test5(id);
    function_bind_test6(id);

    async_test0(id);
    async_test1(id);
    async_test2(id);
    async_test3(id);
}

void run_local_tests()
{
    member_bind_test0();
    member_bind_test1();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    // run the test on all localities
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    for (hpx::naming::id_type const& id : localities)
        run_tests(id);

    // run local tests
    run_local_tests();

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

