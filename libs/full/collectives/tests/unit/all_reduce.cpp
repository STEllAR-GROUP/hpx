//  Copyright (c) 2019-2023 Hartmut Kaiser
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

constexpr char const* all_reduce_direct_basename = "/test/all_reduce_direct/";

void test_one_shot_use()
{
    std::uint32_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t const here = hpx::get_locality_id();

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result =
            all_reduce(all_reduce_direct_basename, value,
                std::plus<std::uint32_t>{}, num_sites_arg(num_localities),
                this_site_arg(here), generation_arg(i + 1));

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

void test_multiple_use()
{
    std::uint32_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t const here = hpx::get_locality_id();

    auto const all_reduce_direct_client =
        create_communicator(all_reduce_direct_basename,
            num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result = all_reduce(
            all_reduce_direct_client, value, std::plus<std::uint32_t>{});

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

void test_multiple_use_with_generation()
{
    std::uint32_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t const here = hpx::get_locality_id();

    auto const all_reduce_direct_client =
        create_communicator(all_reduce_direct_basename,
            num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = here;

        hpx::future<std::uint32_t> overall_result =
            all_reduce(all_reduce_direct_client, value,
                std::plus<std::uint32_t>{}, generation_arg(i + 1));

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

void test_local_use()
{
    constexpr std::uint32_t num_sites = 10;

    std::vector<hpx::future<void>> sites;
    sites.reserve(num_sites);

    // launch num_sites threads to represent different sites
    for (std::uint32_t site = 0; site != num_sites; ++site)
    {
        sites.push_back(hpx::async([site]() {
            auto const all_reduce_direct_client =
                create_local_communicator(all_reduce_direct_basename,
                    num_sites_arg(num_sites), this_site_arg(site));

            // test functionality based on immediate local result value
            auto value = site;

            hpx::future<std::uint32_t> result =
                all_reduce(all_reduce_direct_client, value, std::plus<>{},
                    this_site_arg(site));

            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != 10; ++j)
            {
                sum += j;
            }

            HPX_TEST_EQ(sum, result.get());
        }));
    }

    hpx::wait_all(std::move(sites));
}

int hpx_main()
{
    test_one_shot_use();
    test_multiple_use();
    test_multiple_use_with_generation();

    test_local_use();

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
