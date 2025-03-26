//  Copyright (c) 2025 Michael Ferguson
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

int hpx_main()
{
    // this is run on all localities (nodes)
    auto const rank = hpx::get_locality_id();
    auto const num_ranks = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_EQ(num_ranks, static_cast<std::uint32_t>(2));

    std::int64_t const total_here = rank + 100;

    hpx::future<std::int64_t> f_start = hpx::collectives::exclusive_scan(
        "my_scan", total_here, std::plus<std::int64_t>{},
        hpx::collectives::num_sites_arg(), hpx::collectives::this_site_arg(),
        hpx::collectives::generation_arg(1));

    std::int64_t const start = f_start.get();
    if (rank == 0)
    {
        HPX_TEST_EQ(start, 0);    // default constructed
    }
    else
    {
        HPX_TEST_EQ(start, 100);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // run hpx_main on all localities
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
