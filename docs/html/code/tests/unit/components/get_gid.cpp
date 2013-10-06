////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>

using hpx::components::stub_base;
using hpx::components::client_base;
using hpx::components::managed_component;
using hpx::components::managed_component_base;

using hpx::find_here;
using hpx::async;

struct test_server : managed_component_base<test_server>
{
    void check_gid() const
    {
        hpx::id_type id = get_gid();
        HPX_TEST_NEQ(hpx::invalid_id, id);
    }

    HPX_DEFINE_COMPONENT_CONST_ACTION(test_server, check_gid, check_gid_action);
};

typedef managed_component<test_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, test_server);

typedef test_server::check_gid_action check_gid_action;
HPX_REGISTER_ACTION_DECLARATION(check_gid_action);
HPX_REGISTER_ACTION(check_gid_action);

struct test_client : client_base<test_client, stub_base<test_server> >
{
    void check_gid() { async<check_gid_action>(this->get_gid()).get(); }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_client t;

    t.create(find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_gid());

    t.check_gid();

    return hpx::util::report_errors();
}

