//  Copyright (c) 2015-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
static const char* const latch_name = "latch_remote_test";

///////////////////////////////////////////////////////////////////////////////
hpx::distributed::latch create_latch(
    std::size_t num_threads, std::size_t generation)
{
    std::string name(latch_name);
    name += std::to_string(generation);

    hpx::distributed::latch l;
    if (hpx::get_locality_id() == 0)
    {
        // Create the latch on locality zero, let it synchronize as many
        // threads as we have localities.
        l = hpx::distributed::latch(num_threads);

        // Register the new instance so that the other localities can connect
        // to it.
        l.register_as(name);
    }
    else
    {
        // Connect to the latch created on locality zero.
        l.connect_to(name);
    }
    return l;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    // count_down_and_wait
    {
        hpx::distributed::latch l = create_latch(num_localities, 0);
        HPX_TEST(!l.try_wait());

        // Wait for all localities to reach this point.
        l.arrive_and_wait();

        HPX_TEST(l.is_ready());
    }

    // count_down/wait
    {
        hpx::distributed::latch l = create_latch(num_localities, 1);
        HPX_TEST(!l.try_wait());

        // Wait for all localities to reach this point.
        if (hpx::get_locality_id() == 0)
        {
            l.arrive_and_wait();
            HPX_TEST(l.try_wait());
        }
        else
        {
            l.count_down(1);
            l.wait();
            HPX_TEST(l.try_wait());
        }
    }

    HPX_TEST_EQ(hpx::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // make sure hpx_main will run on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
#endif
