//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/traits.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct decrement_server
  : hpx::components::managed_component_base<decrement_server>
{
    hpx::future<std::int32_t> call(std::int32_t i) const
    {
        return hpx::make_ready_future(i - 1);
    }

    HPX_DEFINE_COMPONENT_ACTION(decrement_server, call);
};

typedef hpx::components::managed_component<decrement_server> server_type;
HPX_REGISTER_COMPONENT(server_type, decrement_server);

typedef decrement_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::simple_component_base<test_server>
{
    hpx::future<std::int32_t> increment(std::int32_t i)
    {
        return hpx::make_ready_future(i + 1);
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, increment);

    hpx::future<std::int32_t> increment_with_future(
        hpx::shared_future<std::int32_t> fi)
    {
        return hpx::make_ready_future(fi.get() + 1);
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server, increment_with_future);
};

typedef hpx::components::simple_component<test_server> test_server_type;
HPX_REGISTER_COMPONENT(test_server_type, test_server);

typedef test_server::increment_action increment_action;
HPX_REGISTER_ACTION_DECLARATION(increment_action);
HPX_REGISTER_ACTION(increment_action);

typedef test_server::increment_with_future_action increment_with_future_action;
HPX_REGISTER_ACTION_DECLARATION(increment_with_future_action);
HPX_REGISTER_ACTION(increment_with_future_action);

struct test_client : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::id_type const& id)
      : base_type(id)
    {
    }
    test_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }
};

///////////////////////////////////////////////////////////////////////////////
void test_remote_async_unwrap_result(test_client const& target)
{
    {
        increment_action inc;

        hpx::future<std::int32_t> f1 =
            hpx::async(inc, hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<std::int32_t> f3 =
            hpx::async(hpx::launch::sync, inc, hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f3.get(), 43);
    }

    {
        increment_with_future_action inc;

        hpx::lcos::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        hpx::future<std::int32_t> f1 =
            hpx::async(inc, hpx::unwrap_result(target), f);
        hpx::future<std::int32_t> f2 =
            hpx::async(hpx::launch::all, inc, hpx::unwrap_result(target), f);

        p.set_value(42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<std::int32_t> f3 =
            hpx::async(hpx::launch::sync, inc, hpx::unwrap_result(target), f);

        HPX_TEST_EQ(f3.get(), 43);
    }

    {
        hpx::future<std::int32_t> f1 =
            hpx::async<increment_action>(hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f1.get(), 43);

        hpx::future<std::int32_t> f2 = hpx::async<increment_action>(
            hpx::launch::all, hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f2.get(), 43);

        hpx::future<std::int32_t> f3 = hpx::async<increment_action>(
            hpx::launch::sync, hpx::unwrap_result(target), 42);
        HPX_TEST_EQ(f3.get(), 43);
    }
}

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& id : localities)
    {
        test_client client(hpx::new_<test_client>(id));
        test_remote_async_unwrap_result(client);
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
