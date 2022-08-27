//  Copyright (c) 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    using base_type = hpx::components::component_base<test_server>;

    test_server() = default;
    ~test_server() = default;

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // components which should be copied using hpx::copy<> need to
    // be Serializable and CopyConstructable. In the remote case
    // it can be MoveConstructable in which case the serialized data
    // is moved into the components constructor.
    test_server(test_server const& rhs) = default;
    test_server(test_server&& rhs) = default;

    test_server& operator=(test_server const&)
    {
        return *this;
    }
    test_server& operator=(test_server&&)
    {
        return *this;
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action)

    template <typename Archive>
    void serialize(Archive&, unsigned)
    {
    }
};

using server_type = hpx::components::component<test_server>;
HPX_REGISTER_COMPONENT(server_type, test_server)

using call_action = test_server::call_action;
HPX_REGISTER_ACTION(call_action)

struct test_client : hpx::components::client_base<test_client, test_server>
{
    using base_type = hpx::components::client_base<test_client, test_server>;

    test_client() = default;
    test_client(hpx::shared_future<hpx::id_type> const& id)
      : base_type(id)
    {
    }
    test_client(hpx::id_type&& id)
      : base_type(std::move(id))
    {
    }

    hpx::id_type call() const
    {
        return call_action()(this->get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(id);
    HPX_TEST_NEQ(hpx::invalid_id, t1.get_id());

    try
    {
        // create a copy of t1 on same locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id()));
        HPX_TEST_NEQ(hpx::invalid_id, t2.get_id());

        // the new object should life on id
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&)
    {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_here(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(id);
    HPX_TEST_NEQ(hpx::invalid_id, t1.get_id());

    try
    {
        // create a copy of t1 here
        test_client t2(
            hpx::components::copy<test_server>(t1.get_id(), hpx::find_here()));
        HPX_TEST_NEQ(hpx::invalid_id, t2.get_id());

        // the new object should life here
        HPX_TEST_EQ(t2.call(), hpx::find_here());

        return true;
    }
    catch (hpx::exception const&)
    {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_there(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(hpx::find_here());
    HPX_TEST_NEQ(hpx::invalid_id, t1.get_id());

    try
    {
        // create a copy of t1 on given locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id(), id));
        HPX_TEST_NEQ(hpx::invalid_id, t2.get_id());

        // the new object should life there
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&)
    {
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
#endif
