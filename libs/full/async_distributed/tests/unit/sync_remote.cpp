//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/sync.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::int32_t increment(std::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);

std::int32_t increment_with_future(hpx::shared_future<std::int32_t> fi)
{
    return fi.get() + 1;
}
HPX_PLAIN_ACTION(increment_with_future);

///////////////////////////////////////////////////////////////////////////////
struct decrement_server
  : hpx::components::managed_component_base<decrement_server>
{
    std::int32_t call(std::int32_t i) const
    {
        return i - 1;
    }

    HPX_DEFINE_COMPONENT_ACTION(decrement_server, call);
};

typedef hpx::components::managed_component<decrement_server> server_type;
HPX_REGISTER_COMPONENT(server_type, decrement_server);

typedef decrement_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
void test_remote_sync(hpx::id_type const& target)
{
    {
        increment_action inc;

        std::int32_t r1 = hpx::sync(inc, target, 42);
        HPX_TEST_EQ(r1, 43);

        std::int32_t r2 = hpx::sync(hpx::launch::all, inc, target, 42);
        HPX_TEST_EQ(r2, 43);

        std::int32_t r3 = hpx::sync(hpx::launch::sync, inc, target, 42);
        HPX_TEST_EQ(r3, 43);
    }

    {
        increment_with_future_action inc;
        hpx::lcos::promise<std::int32_t> p;
        hpx::shared_future<std::int32_t> f = p.get_future();

        p.set_value(42);

        std::int32_t r1 = hpx::sync(inc, target, f);
        std::int32_t r2 = hpx::sync(hpx::launch::all, inc, target, f);

        HPX_TEST_EQ(r1, 43);
        HPX_TEST_EQ(r2, 43);

        std::int32_t r3 = hpx::sync(hpx::launch::sync, inc, target, f);
        HPX_TEST_EQ(r3, 43);
    }

    {
        increment_action inc;

        std::int32_t r1 = hpx::sync(hpx::util::bind(inc, target, 42));
        HPX_TEST_EQ(r1, 43);
    }

    {
        std::int32_t r1 = hpx::sync<increment_action>(target, 42);
        HPX_TEST_EQ(r1, 43);

        std::int32_t r2 =
            hpx::sync<increment_action>(hpx::launch::all, target, 42);
        HPX_TEST_EQ(r2, 43);

        std::int32_t r3 =
            hpx::sync<increment_action>(hpx::launch::sync, target, 42);
        HPX_TEST_EQ(r3, 43);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(target);
        hpx::id_type dec = dec_f.get();

        call_action call;

        std::int32_t r1 = hpx::sync(call, dec, 42);
        HPX_TEST_EQ(r1, 41);

        std::int32_t r2 = hpx::sync(hpx::launch::all, call, dec, 42);
        HPX_TEST_EQ(r2, 41);

        std::int32_t r3 = hpx::sync(hpx::launch::sync, call, dec, 42);
        HPX_TEST_EQ(r3, 41);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(target);
        hpx::id_type dec = dec_f.get();

        call_action call;

        std::int32_t r1 = hpx::sync(hpx::util::bind(call, dec, 42));
        HPX_TEST_EQ(r1, 41);

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;

        std::int32_t r2 = hpx::sync(hpx::util::bind(call, _1, 42), dec);
        HPX_TEST_EQ(r2, 41);

        std::int32_t r3 = hpx::sync(hpx::util::bind(call, _1, _2), dec, 42);
        HPX_TEST_EQ(r3, 41);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(target);
        hpx::id_type dec = dec_f.get();

        std::int32_t r1 = hpx::sync<call_action>(dec, 42);
        HPX_TEST_EQ(r1, 41);

        std::int32_t r2 = hpx::sync<call_action>(hpx::launch::all, dec, 42);
        HPX_TEST_EQ(r2, 41);

        std::int32_t r3 = hpx::sync<call_action>(hpx::launch::sync, dec, 42);
        HPX_TEST_EQ(r3, 41);
    }

    {
        increment_with_future_action inc;
        hpx::shared_future<std::int32_t> f =
            hpx::async(hpx::launch::deferred, hpx::util::bind(&increment, 42));

        std::int32_t r1 = hpx::sync(inc, target, f);
        std::int32_t r2 = hpx::sync(hpx::launch::all, inc, target, f);
        std::int32_t r3 = hpx::sync(hpx::launch::sync, inc, target, f);

        HPX_TEST_EQ(r1, 44);
        HPX_TEST_EQ(r2, 44);
        HPX_TEST_EQ(r3, 44);
    }

    {
        auto policy1 =
            hpx::launch::select([]() { return hpx::launch::deferred; });

        increment_with_future_action inc;
        hpx::shared_future<std::int32_t> f =
            hpx::async(policy1, hpx::util::bind(&increment, 42));

        std::atomic<int> count(0);
        auto policy2 = hpx::launch::select([&count]() -> hpx::launch {
            if (count++ == 0)
                return hpx::launch::sync;
            return hpx::launch::sync;
        });

        std::int32_t r1 = hpx::sync(policy2, inc, target, f);
        std::int32_t r2 = hpx::sync(policy2, inc, target, f);

        HPX_TEST_EQ(r1, 44);
        HPX_TEST_EQ(r2, 44);
    }
}

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& id : localities)
    {
        test_remote_sync(id);
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
