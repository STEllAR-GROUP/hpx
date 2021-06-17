//  Copyright (c) 2020-2021 Hartmut Kaiser
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

using namespace hpx::collectives;

constexpr char const* gather_direct_basename = "/test/gather_direct/";

void test_one_shot_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    // test functionality based on immediate local result value
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (this_locality == 0)
        {
            std::uint32_t value = 42;
            hpx::future<std::vector<std::uint32_t>> overall_result =
                gather_here(gather_direct_basename, value,
                    num_sites_arg(num_localities), this_site_arg(this_locality),
                    generation_arg(i));

            std::vector<std::uint32_t> sol = overall_result.get();
            for (std::size_t j = 0; j != sol.size(); ++j)
            {
                HPX_TEST(j + 42 == sol[j]);
            }
        }
        else
        {
            hpx::future<void> overall_result =
                gather_there(gather_direct_basename, this_locality + 42,
                    this_site_arg(this_locality), generation_arg(i));
            overall_result.get();
        }
    }
}

void test_multiple_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    // test functionality based on immediate local result value
    auto gather_direct_client = create_communicator(gather_direct_basename,
        num_sites_arg(num_localities), this_site_arg(this_locality));

    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (this_locality == 0)
        {
            hpx::future<std::vector<std::uint32_t>> overall_result =
                gather_here(gather_direct_client, std::uint32_t(42));

            std::vector<std::uint32_t> sol = overall_result.get();
            for (std::size_t j = 0; j != sol.size(); ++j)
            {
                HPX_TEST(j + 42 == sol[j]);
            }
        }
        else
        {
            hpx::future<void> overall_result =
                gather_there(gather_direct_client, this_locality + 42);
            overall_result.get();
        }
    }
}

int hpx_main()
{
    test_one_shot_use();
    test_multiple_use();

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
