//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    test_server() {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // components which should be copied using hpx::copy<> need to
    // be Serializable and need to have a special constructor
    test_server(boost::shared_ptr<test_server> rhs) {}

    template <typename Archive>
    void serialize(Archive&ar, unsigned version) {}

    HPX_DEFINE_COMPONENT_CONST_ACTION(test_server, call, call_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}
    test_client(hpx::future<hpx::id_type> const& id) : base_type(id) {}

    hpx::id_type call() const { return call_action()(this->get_gid()); }
};

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component(hpx::id_type id)
{
    // create component on given locality
    test_client t1;
    t1.create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_gid());

    try {
        // create a copy of t1 on same locality
        test_client t2(hpx::components::copy<test_server>(t1.get_gid()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_gid());

        // the new object should life on id
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_here(hpx::id_type id)
{
    // create component on given locality
    test_client t1;
    t1.create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_gid());

    try {
        // create a copy of t1 here
        test_client t2(hpx::components::copy<test_server>(t1.get_gid(), hpx::find_here()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_gid());

        // the new object should life here
        HPX_TEST_EQ(t2.call(), hpx::find_here());

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_there(hpx::id_type id)
{
    // create component on given locality
    test_client t1;
    t1.create(hpx::find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_gid());

    try {
        // create a copy of t1 on given locality
        test_client t2(hpx::components::copy<test_server>(t1.get_gid(), id));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_gid());

        // the new object should life there
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_TEST_EQ(e.get_error(), hpx::bad_parameter);
    }

    return false;
}

int main()
{
    HPX_TEST(test_copy_component(hpx::find_here()));
    HPX_TEST(test_copy_component_here(hpx::find_here()));
    HPX_TEST(test_copy_component_there(hpx::find_here()));

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    BOOST_FOREACH(hpx::id_type const& id, localities)
    {
        HPX_TEST(test_copy_component(id));
        HPX_TEST(test_copy_component_here(id));
        HPX_TEST(test_copy_component_there(id));
    }

    return hpx::util::report_errors();
}

