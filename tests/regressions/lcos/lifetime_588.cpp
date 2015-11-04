//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #588: Continuations do not
// keep object alive

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    ~test_server()
    {
    }

    hpx::id_type create_new(hpx::id_type const& id) const;

    HPX_DEFINE_COMPONENT_ACTION(test_server, create_new, create_new_action);
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::create_new_action create_new_action;
HPX_REGISTER_ACTION_DECLARATION(create_new_action);
HPX_REGISTER_ACTION(create_new_action);

struct test_client
  : hpx::components::client_base<
        test_client, hpx::components::stub_base<test_server> >
{
    typedef hpx::components::client_base<
        test_client, hpx::components::stub_base<test_server> >
    client_base_type;

    // create a new instance of a test_server
    test_client()
    {}

    // initialize the client from a given server instance
    explicit test_client(hpx::id_type const& id)
      : client_base_type(id)
    {}
    explicit test_client(hpx::shared_future<hpx::id_type> const& fgid)
      : client_base_type(fgid)
    {}

    test_client create_new(hpx::id_type const& id) const
    {
        create_new_action new_;
        return test_client(hpx::async(new_, this->get_id(), id));
    }
};

// ask the server to create a new instance
hpx::id_type test_server::create_new(hpx::id_type const& id) const
{
    // this waits for the new object to be created
    return test_client::create(id).get_id();
}

int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    for (hpx::id_type const& id : localities)
    {
        // repeating this a couple of times forces the issue ...
        for (int i = 0; i != 100; ++i)
        {
            test_client c =  test_client::create(id);  // create a new instance

            // this construct overwrites the original client with a newly created
            // one which causes the only reference to the initial test_server to go
            // out of scope too early.
            c = c.create_new(id);

            // the new instance 'c' goes out of scope here, which makes the future
            // holding the only reference to the second server instance disappear
        }
    }

    return hpx::util::report_errors();
}
