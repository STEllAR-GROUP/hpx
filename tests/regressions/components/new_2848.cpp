//  Copyright (c) 2015-2017 Hartmut Kaiser
//  Copyright (c) 2017 Igor Krivenko
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
struct test_server : hpx::components::simple_component_base<test_server>
{
    test_server() = delete;
    test_server(int i) : i_(i) {}

    hpx::id_type call() const { return hpx::find_here(); }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call);

    int i_;
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
    // make sure created objects live on locality they are supposed to be
    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        hpx::id_type id = hpx::new_<test_server>(loc, 42).get();
        HPX_TEST(hpx::async<call_action>(id).get() == loc);
    }

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        test_client t1 = hpx::new_<test_client>(loc, 42);
        HPX_TEST(t1.call() == loc);
    }

    // make sure distribution policy is properly used
    hpx::id_type id = hpx::new_<test_server>(hpx::default_layout, 42).get();
    HPX_TEST(hpx::async<call_action>(id).get() == hpx::find_here());

    test_client t2 = hpx::new_<test_client>(hpx::default_layout, 42);
    HPX_TEST(t2.call() == hpx::find_here());

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        hpx::id_type id =
            hpx::new_<test_server>(hpx::default_layout(loc), 42).get();
        HPX_TEST(hpx::async<call_action>(id).get() == loc);
    }

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        test_client t3 = hpx::new_<test_client>(hpx::default_layout(loc), 42);
        HPX_TEST(t3.call() == loc);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_create_multiple_instances()
{
    // make sure created objects live on locality they are supposed to be
    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        std::vector<hpx::id_type> ids =
            hpx::new_<test_server[]>(loc, 10, 42).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (hpx::id_type const& id: ids)
        {
            HPX_TEST(hpx::async<call_action>(id).get() == loc);
        }
    }

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        std::vector<test_client> ids =
            hpx::new_<test_client[]>(loc, 10, 42).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (test_client const& c: ids)
        {
            HPX_TEST(c.call() == loc);
        }
    }

    // make sure distribution policy is properly used
    std::vector<hpx::id_type> ids =
        hpx::new_<test_server[]>(hpx::default_layout, 10, 42).get();
    HPX_TEST_EQ(ids.size(), std::size_t(10));
    for (hpx::id_type const& id: ids)
    {
        HPX_TEST(hpx::async<call_action>(id).get() == hpx::find_here());
    }

    std::vector<test_client> clients =
        hpx::new_<test_client[]>(hpx::default_layout, 10, 42).get();
    HPX_TEST_EQ(clients.size(), std::size_t(10));
    for (test_client const& c: clients)
    {
        HPX_TEST(c.call() == hpx::find_here());
    }

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        std::vector<hpx::id_type> ids =
            hpx::new_<test_server[]>(hpx::default_layout(loc), 10, 42).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (hpx::id_type const& id: ids)
        {
            HPX_TEST(hpx::async<call_action>(id).get() == loc);
        }
    }

    for (hpx::id_type const& loc: hpx::find_all_localities())
    {
        std::vector<test_client> ids =
            hpx::new_<test_client[]>(hpx::default_layout(loc), 10, 42).get();
        HPX_TEST_EQ(ids.size(), std::size_t(10));

        for (test_client const& c: ids)
        {
            HPX_TEST(c.call() == loc);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_create_single_instance();
    test_create_multiple_instances();

    return 0;
}

