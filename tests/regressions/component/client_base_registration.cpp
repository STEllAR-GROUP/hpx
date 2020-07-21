//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Client_base can be used to register a component instance with AGAS. This
// test verifies that the component instance does not get unregistered anymore
// once the client instance that was registering the component goes out of
// scope.

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::simple_component_base<test_server>
{
    test_server() = delete;
    explicit test_server(int i) : i_(i) {}

    int call() const { return i_; }

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

    test_client() = default;

    test_client(hpx::id_type const& id)
      : base_type(id)
    {}
    explicit test_client(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    int call() const
    {
        return hpx::async<call_action>(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_client_registration()
{
    test_client c_outer;

    {
        // create component instance
        test_client c = hpx::new_<test_client>(hpx::find_here(), 42).get();
        c.register_as("test-client-base");

        HPX_TEST_EQ(42, c.call());

        c_outer = c;        // should keep instance alive
    }

    {
        test_client c;
        c.connect_to("test-client-base");

        HPX_TEST_EQ(42, c.call());
    }

    (void) c_outer;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_client_registration();

    return 0;
}

