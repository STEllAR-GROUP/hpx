//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct A
{
    A() = default;

    HPX_NON_COPYABLE(A);
};

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::simple_component_base<test_server>
{
    test_server() = default;
    test_server(A const&) {}

    hpx::id_type call() const { return hpx::find_here(); }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

struct test_client : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::id_type const& id)
      : base_type(id)
    {}
    test_client(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    hpx::id_type call() const
    {
        return hpx::async<call_action>(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_create_single_instance()
{
    hpx::id_type id = hpx::local_new<test_server>().get();
    HPX_TEST(hpx::async<call_action>(id).get() == hpx::find_here());

    test_client t1 = hpx::local_new<test_client>();
    HPX_TEST(t1.call() == hpx::find_here());
}

void test_create_single_instance_non_copyable_arg()
{
    A a;

    hpx::id_type id = hpx::local_new<test_server>(a).get();
    HPX_TEST(hpx::async<call_action>(id).get() == hpx::find_here());

    test_client t1 = hpx::local_new<test_client>(a);
    HPX_TEST(t1.call() == hpx::find_here());
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_create_single_instance();
    test_create_single_instance_non_copyable_arg();

    return 0;
}

