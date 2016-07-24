//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Demonstrate the use of hpx::lcos::latch

#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
static const char* const latch_name = "latch_remote_example";

int hpx_main()
{
    hpx::lcos::latch l;
    if (hpx::get_locality_id() == 0)
    {
        // Create the latch on locality zero, let it synchronize as many
        // threads as we have localities.
        l = hpx::lcos::latch(hpx::get_num_localities_sync());

        // Register the new instance so that the other localities can connect
        // to it.
        l.register_as(latch_name);
    }
    else
    {
        // Connect to the latch created on locality zero.
        l.connect_to(latch_name);
    }

    // Wait for all localities to reach this point.
    l.count_down_and_wait();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // make sure hpx_main will run on all localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    return hpx::init(argc, argv, cfg);
}
