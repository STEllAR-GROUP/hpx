//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct A
{
    A() = default;
    A(A const&) = delete;
    A& operator=(A const&) = delete;
};

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    test_server() = default;
    test_server(A const&) {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call);
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

struct test_client : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::id_type const& id)
      : base_type(id)
    {
    }
    test_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    hpx::id_type call() const
    {
        return hpx::async<call_action>(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_create_single_instance()
{
    hpx::id_type id = hpx::local_new<test_server>().get();
    HPX_TEST_EQ(hpx::async<call_action>(id).get(), hpx::find_here());

    hpx::id_type id1 = hpx::local_new<test_server>(hpx::launch::sync);
    HPX_TEST_EQ(hpx::async<call_action>(id1).get(), hpx::find_here());

    test_client t1 = hpx::local_new<test_client>();
    HPX_TEST_EQ(t1.call(), hpx::find_here());
}

void test_create_single_instance_non_copyable_arg()
{
    A a;

    hpx::id_type id = hpx::local_new<test_server>(a).get();
    HPX_TEST_EQ(hpx::async<call_action>(id).get(), hpx::find_here());

    hpx::id_type id1 = hpx::local_new<test_server>(hpx::launch::sync, a);
    HPX_TEST_EQ(hpx::async<call_action>(id1).get(), hpx::find_here());

    test_client t1 = hpx::local_new<test_client>(a);
    HPX_TEST_EQ(t1.call(), hpx::find_here());
}

///////////////////////////////////////////////////////////////////////////////
void test_create_multiple_instances()
{
    // make sure created objects live on locality they are supposed to be
    {
        std::vector<hpx::id_type> ids = hpx::local_new<test_server[]>(10).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (hpx::id_type const& id : ids)
        {
            HPX_TEST_EQ(hpx::async<call_action>(id).get(), hpx::find_here());
        }
    }

    {
        std::vector<hpx::id_type> ids =
            hpx::local_new<test_server[]>(hpx::launch::sync, 10);
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (hpx::id_type const& id : ids)
        {
            HPX_TEST_EQ(hpx::async<call_action>(id).get(), hpx::find_here());
        }
    }

    {
        std::vector<test_client> ids = hpx::local_new<test_client[]>(10).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (test_client const& c : ids)
        {
            HPX_TEST_EQ(c.call(), hpx::find_here());
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_create_single_instance();
    test_create_single_instance_non_copyable_arg();

    test_create_multiple_instances();

    return 0;
}
#endif
