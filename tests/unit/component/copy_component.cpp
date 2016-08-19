//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    typedef hpx::components::simple_component_base<test_server> base_type;

    test_server() {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // components which should be copied using hpx::copy<> need to
    // be Serializable and CopyConstructable. In the remote case
    // it can be MoveConstructable in which case the serialized data
    // is moved into the components constructor.
    test_server(test_server const& rhs)
      : base_type(rhs)
    {}

    test_server(test_server && rhs)
      : base_type(std::move(rhs))
    {}

    test_server& operator=(test_server const &) { return *this; }
    test_server& operator=(test_server &&) { return *this; }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);

    template <typename Archive>
    void serialize(Archive&ar, unsigned version) {}

private:
    ;
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION(call_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

    hpx::id_type call() const { return call_action()(this->get_id()); }
};

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 on same locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life on id
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_here(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 here
        test_client t2(hpx::components::copy<test_server>(
            t1.get_id(), hpx::find_here()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life here
        HPX_TEST_EQ(t2.call(), hpx::find_here());

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_there(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(hpx::find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 on given locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id(), id));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life there
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

int main()
{
    HPX_TEST(test_copy_component(hpx::find_here()));
    HPX_TEST(test_copy_component_here(hpx::find_here()));
    HPX_TEST(test_copy_component_there(hpx::find_here()));

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(test_copy_component(id));
        HPX_TEST(test_copy_component_here(id));
        HPX_TEST(test_copy_component_there(id));
    }

    return hpx::util::report_errors();
}

