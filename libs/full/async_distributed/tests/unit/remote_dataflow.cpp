//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::uint32_t> void_f_count;

void void_f(hpx::future<int>&&)
{
    ++void_f_count;
}
HPX_PLAIN_ACTION(void_f);

hpx::id_type id_f()
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(id_f);

void plain_actions(hpx::id_type const& there)
{
    hpx::id_type here = hpx::find_here();

    {
        void_f_count.store(0);

        hpx::future<void> f1 =
            hpx::dataflow(void_f_action(), here, hpx::make_ready_future(42));
        hpx::future<void> f2 =
            hpx::dataflow<void_f_action>(here, hpx::make_ready_future(42));

        f1.get();
        f2.get();

        HPX_TEST_EQ(void_f_count, 2u);
    }

    {
        hpx::future<hpx::id_type> f1 = hpx::dataflow(id_f_action(), there);
        hpx::future<hpx::id_type> f2 = hpx::dataflow<id_f_action>(there);

        HPX_TEST_EQ(there, f1.get());
        HPX_TEST_EQ(there, f2.get());
    }

    hpx::launch policies[] = {
        hpx::launch::async, hpx::launch::deferred, hpx::launch::sync};

    for (int i = 0; i != 2; ++i)
    {
        {
            void_f_count.store(0);

            hpx::future<void> f1 = hpx::dataflow(
                policies[i], void_f_action(), here, hpx::make_ready_future(42));
            hpx::future<void> f2 = hpx::dataflow<void_f_action>(
                policies[i], here, hpx::make_ready_future(42));

            f1.get();
            f2.get();

            HPX_TEST_EQ(void_f_count, 2u);
        }

        {
            hpx::future<hpx::id_type> f1 =
                hpx::dataflow(policies[i], id_f_action(), there);
            hpx::future<hpx::id_type> f2 =
                hpx::dataflow<id_f_action>(policies[i], there);

            HPX_TEST_EQ(there, f1.get());
            HPX_TEST_EQ(there, f2.get());
        }
    }

    auto policy1 = hpx::launch::select([]() { return hpx::launch::sync; });

    {
        void_f_count.store(0);

        hpx::future<void> f1 = hpx::dataflow(
            policy1, void_f_action(), here, hpx::make_ready_future(42));
        hpx::future<void> f2 = hpx::dataflow<void_f_action>(
            policy1, here, hpx::make_ready_future(42));

        f1.get();
        f2.get();

        HPX_TEST_EQ(void_f_count, 2u);
    }

    std::atomic<int> count(0);
    auto policy2 = hpx::launch::select([&count]() -> hpx::launch {
        if (count++ == 0)
            return hpx::launch::async;
        return hpx::launch::sync;
    });

    {
        hpx::future<hpx::id_type> f1 =
            hpx::dataflow(policy2, id_f_action(), there);
        hpx::future<hpx::id_type> f2 =
            hpx::dataflow<id_f_action>(policy2, there);

        HPX_TEST_EQ(there, f1.get());
        HPX_TEST_EQ(there, f2.get());
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for (hpx::id_type const& id : hpx::find_all_localities())
    {
        plain_actions(id);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}
#endif
