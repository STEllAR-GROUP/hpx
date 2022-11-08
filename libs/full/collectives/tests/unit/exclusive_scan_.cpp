//  Copyright (c) 2019-2022 Hartmut Kaiser
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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

constexpr char const* exclusive_scan_basename = "/test/exclusive_scan/";

void test_one_shot_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result =
            exclusive_scan(exclusive_scan_basename, value,
                std::plus<std::uint32_t>{}, num_sites_arg(num_localities),
                this_site_arg(here), generation_arg(i + 1));

        if (here == 0)
        {
            // root_site is special
            HPX_TEST_EQ(std::uint32_t(0), overall_result.get());
        }
        else
        {
            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != value; ++j)
            {
                sum += j;
            }
            HPX_TEST_EQ(sum, overall_result.get());
        }
    }
}

void test_multiple_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    auto exclusive_scan_client = create_communicator(exclusive_scan_basename,
        num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result = exclusive_scan(
            exclusive_scan_client, value, std::plus<std::uint32_t>{});

        if (here == 0)
        {
            // root_site is special
            HPX_TEST_EQ(std::uint32_t(0), overall_result.get());
        }
        else
        {
            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != value; ++j)
            {
                sum += j;
            }
            HPX_TEST_EQ(sum, overall_result.get());
        }
    }
}

void test_multiple_use_with_generation()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    auto exclusive_scan_client = create_communicator(exclusive_scan_basename,
        num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result =
            exclusive_scan(exclusive_scan_client, value,
                std::plus<std::uint32_t>{}, generation_arg(i + 1));

        if (here == 0)
        {
            // root_site is special
            HPX_TEST_EQ(std::uint32_t(0), overall_result.get());
        }
        else
        {
            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != value; ++j)
            {
                sum += j;
            }
            HPX_TEST_EQ(sum, overall_result.get());
        }
    }
}

int hpx_main()
{
    test_one_shot_use();
    test_multiple_use();
    test_multiple_use_with_generation();

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
