//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::id_type test_colocation()
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(test_colocation);

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::managed_component_base<test_server>
{
    hpx::id_type call() const
    {
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);
};

typedef hpx::components::managed_component<test_server> server_type;
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
void test(hpx::id_type there)
{
    test_client t1 = test_client::create(there);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), there);

    // verify for remote component
    HPX_TEST_EQ(hpx::get_colocation_id_sync(t1.get_id()), there);

    HPX_TEST_EQ(hpx::async<test_colocation_action>(
        hpx::colocated(t1.get_id())).get(), there);

    test_colocation_action act;
    HPX_TEST_EQ(hpx::async(act, hpx::colocated(t1.get_id())).get(), there);

    // verify for remote locality
    HPX_TEST_EQ(hpx::get_colocation_id_sync(there), there);

    HPX_TEST_EQ(hpx::async<test_colocation_action>(
        hpx::colocated(there)).get(), there);

    HPX_TEST_EQ(hpx::async(act, hpx::colocated(there)).get(), there);
}

int hpx_main()
{
    for (hpx::id_type const& id : hpx::find_all_localities())
    {
        test(id);
    }

    bool caught_exception = false;
    try {
        hpx::get_colocation_id_sync(hpx::invalid_id);
        HPX_TEST(false);
    }
    catch (hpx::exception const&) {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

