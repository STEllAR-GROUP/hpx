//  Copyright 2014 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1211:
// Direct actions cause the future's shared_state to be leaked

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/modules/testing.hpp>

struct test_server : hpx::components::simple_component_base<test_server>
{
    static bool destructor_called;

    test_server()
    {
    }

    ~test_server()
    {
        HPX_TEST(!destructor_called);
        destructor_called = true;
    }

    int test() const
    {
        return 42;
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(test_server, test, test_action);
};

bool test_server::destructor_called = false;

typedef hpx::components::simple_component<test_server> test_server_type;
HPX_REGISTER_COMPONENT(test_server_type, test_server);

typedef test_server::test_action test_action;
HPX_REGISTER_ACTION(test_action);

struct test : hpx::components::client_base<test, test_server>
{
    typedef hpx::components::client_base<test, test_server> base_type;

    test(hpx::id_type where)
      : base_type(hpx::new_<test_server>(where))
    {}

    hpx::future<int> call() const
    {
        return hpx::async(test_action(), get_id());
    }
};

int main()
{
    {
        test t(hpx::find_here());
        hpx::future<int> f = t.call();
        HPX_TEST_EQ(f.get(), 42);
    }

    hpx::agas::garbage_collect();   // collects the promise
    hpx::agas::garbage_collect();   // collects the test component instance

    HPX_TEST(test_server::destructor_called);

    return hpx::util::report_errors();
}
#endif
