//  Copyright (c) 2016 Lukas Troska
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2001: Gathering more
// than once segfaults

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

char const* gather_basename = "/test/gather/";

HPX_REGISTER_GATHER(std::uint32_t, test_gather);

int hpx_main()
{
    for (int i = 0; i < 10; ++i)
    {
        hpx::future<std::uint32_t> value =
            hpx::make_ready_future(hpx::get_locality_id());

        if (hpx::get_locality_id() == 0)
        {
            hpx::future<std::vector<std::uint32_t>> overall_result =
                hpx::lcos::gather_here(gather_basename, std::move(value),
                    hpx::get_num_localities(hpx::launch::sync), i);

            std::vector<std::uint32_t> sol = overall_result.get();

            for (std::size_t j = 0; j < sol.size(); ++j)
            {
                HPX_TEST(j == sol[j]);
            }
        }
        else
        {
            hpx::future<void> overall_result =
                hpx::lcos::gather_there(gather_basename, std::move(value), i);

            overall_result.get();
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
