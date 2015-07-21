//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::migration_support<
        hpx::components::simple_component_base<test_server>
    >
{
    test_server() {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized  is moved into the
    // components constructor.
    test_server(test_server const& rhs) {}
    test_server(test_server && rhs) {}

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
HPX_REGISTER_ACTION_DECLARATION(call_action);
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
bool test_migrate_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = test_client::create(source);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);

    try {
        // migrate of t1 to the target
        test_client t2(hpx::components::migrate<test_server>(
            t1.get_id(), target));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should life on the target now
        HPX_TEST_EQ(t2.call(), target);

        return true;
    }
    catch (hpx::exception const&) {
        return false;
    }
}

int main()
{
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(test_migrate_component(hpx::find_here(), id));
        HPX_TEST(test_migrate_component(id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}

