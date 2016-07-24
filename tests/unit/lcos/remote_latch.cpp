//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
static const char* const latch_name = "latch_remote_test";

///////////////////////////////////////////////////////////////////////////////
hpx::lcos::latch create_latch(std::size_t num_threads, std::size_t generation)
{
    std::string name(latch_name);
    name += std::to_string(generation);

    hpx::lcos::latch l;
    if (hpx::get_locality_id() == 0)
    {
        // Create the latch on locality zero, let it synchronize as many
        // threads as we have localities.
        l = hpx::lcos::latch(num_threads);

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
    boost::uint32_t num_localities = hpx::get_num_localities_sync();

    // count_down_and_wait
    {
        hpx::lcos::latch l = create_latch(num_localities, 0);
        HPX_TEST(!l.is_ready());

        // Wait for all localities to reach this point.
        l.count_down_and_wait();

        HPX_TEST(l.is_ready());
    }

    // count_down/wait
    {
        hpx::lcos::latch l = create_latch(num_localities, 1);
        HPX_TEST(!l.is_ready());

        // Wait for all localities to reach this point.
        if (hpx::get_locality_id() == 0)
        {
            l.count_down_and_wait();
            HPX_TEST(l.is_ready());
        }
        else
        {
            l.count_down(1);
            l.wait();
            HPX_TEST(l.is_ready());
        }
    }

    HPX_TEST_EQ(hpx::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // make sure hpx_main will run on all localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
