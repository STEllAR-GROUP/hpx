//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

constexpr char const* gather_basename = "/test/gather/";
constexpr char const* gather_direct_basename = "/test/gather_direct/";

int hpx_main()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    std::uint32_t this_locality = hpx::get_locality_id();

    // test functionality based on future<> of local result
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (this_locality == 0)
        {
            hpx::future<std::vector<std::uint32_t>> overall_result =
                hpx::lcos::gather_here(gather_basename,
                    hpx::make_ready_future(std::uint32_t(42)), num_localities,
                    i);

            std::vector<std::uint32_t> sol = overall_result.get();
            for (std::size_t j = 0; j != sol.size(); ++j)
            {
                HPX_TEST(j + 42 == sol[j]);
            }
        }
        else
        {
            hpx::future<void> overall_result = hpx::lcos::gather_there(
                gather_basename, hpx::make_ready_future(this_locality + 42), i);
            overall_result.get();
        }
    }

    // test functionality based on immediate local result value
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (this_locality == 0)
        {
            hpx::future<std::vector<std::uint32_t>> overall_result =
                hpx::lcos::gather_here(gather_direct_basename,
                    std::uint32_t(42), num_localities, i);

            std::vector<std::uint32_t> sol = overall_result.get();
            for (std::size_t j = 0; j != sol.size(); ++j)
            {
                HPX_TEST(j + 42 == sol[j]);
            }
        }
        else
        {
            hpx::future<void> overall_result = hpx::lcos::gather_there(
                gather_direct_basename, this_locality + 42, i);
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

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}
#endif
