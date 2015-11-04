////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
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
        hpx::id_type id = get_unmanaged_id();
        HPX_TEST_NEQ(hpx::invalid_id, id);
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, check_gid, check_gid_action);
};

typedef managed_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::check_gid_action check_gid_action;
HPX_REGISTER_ACTION_DECLARATION(check_gid_action);
HPX_REGISTER_ACTION(check_gid_action);

struct test_client : client_base<test_client, stub_base<test_server> >
{
    typedef client_base<test_client, stub_base<test_server> > base_type;

    test_client(hpx::future<hpx::id_type>&& id) : base_type(std::move(id)) {}

    void check_gid() { async<check_gid_action>(this->get_id()).get(); }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_client t = test_client::create(find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    t.check_gid();

    return hpx::util::report_errors();
}

