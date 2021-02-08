////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <utility>

using hpx::components::client_base;
using hpx::components::managed_component;
using hpx::components::managed_component_base;
using hpx::components::stub_base;

using hpx::async;
using hpx::find_here;

struct test_server : managed_component_base<test_server>
{
    void check_gid() const
    {
        hpx::id_type id = get_unmanaged_id();
        HPX_TEST_NEQ(hpx::invalid_id, id);
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, check_gid, check_gid_action);
};

using server_type = managed_component<test_server>;
HPX_REGISTER_COMPONENT(server_type, test_server);

using check_gid_action = test_server::check_gid_action;
HPX_REGISTER_ACTION_DECLARATION(check_gid_action);
HPX_REGISTER_ACTION(check_gid_action);

struct test_client : client_base<test_client, stub_base<test_server>>
{
    using base_type = client_base<test_client, stub_base<test_server>>;

    explicit test_client(hpx::id_type const& id) noexcept
      : base_type(id)
    {
    }

    test_client(hpx::future<hpx::id_type>&& id) noexcept
      : base_type(std::move(id))
    {
    }

    void check_gid()
    {
        async<check_gid_action>(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_client t = hpx::new_<test_client>(find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t.get_id());

    t.check_gid();

    return hpx::util::report_errors();
}
#endif
