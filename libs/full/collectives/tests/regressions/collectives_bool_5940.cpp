//  Copyright (c) 2022 Hartmut Kaiser
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

constexpr char const* add_reduce_bool_basename = "/test/add_reduce_bool/";
constexpr char const* broadcast_bool_basename = "/test/broadcast_bool/";
constexpr char const* exclusive_scan_bool_basename =
    "/test/exclusive_scan_bool/";
constexpr char const* inclusive_scan_bool_basename =
    "/test/inclusive_scan_bool/";
constexpr char const* reduce_bool_basename = "/test/reduce_bool/";
constexpr char const* scatter_bool_basename = "/test/scatter_bool/";

void test_all_reduce_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    auto all_reduce_bool_client = create_communicator(add_reduce_bool_basename,
        num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;

        hpx::future<bool> overall_result = all_reduce(all_reduce_bool_client,
            value, std::logical_or<>{}, generation_arg(i + 1));

        HPX_TEST_EQ(value, overall_result.get());
    }
}

void test_broadcast_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(std::uint32_t(2), num_localities);

    std::uint32_t here = hpx::get_locality_id();

    auto broadcast_bool_client = create_communicator(broadcast_bool_basename,
        num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;
        if (here == 0)
        {
            hpx::future<bool> result = broadcast_to(
                broadcast_bool_client, value, generation_arg(i + 1));

            HPX_TEST_EQ(value, result.get());
        }
        else
        {
            hpx::future<bool> result = hpx::collectives::broadcast_from<bool>(
                broadcast_bool_client, generation_arg(i + 1));

            HPX_TEST_EQ(value, result.get());
        }
    }
}

void test_exclusive_scan_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    auto exclusive_scan_bool_client =
        create_communicator(exclusive_scan_bool_basename,
            num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;

        hpx::future<bool> overall_result =
            exclusive_scan(exclusive_scan_bool_client, value,
                std::logical_or<>{}, generation_arg(i + 1));

        HPX_TEST_EQ(value, overall_result.get());
    }
}

void test_inclusive_scan_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t here = hpx::get_locality_id();

    auto inclusive_scan_client =
        create_communicator(inclusive_scan_bool_basename,
            num_sites_arg(num_localities), this_site_arg(here));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;

        hpx::future<bool> overall_result = inclusive_scan(inclusive_scan_client,
            value, std::logical_or<>{}, generation_arg(i + 1));

        HPX_TEST_EQ(value, overall_result.get());
    }
}

void test_reduce_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    auto reduce_bool_client = create_communicator(reduce_bool_basename,
        num_sites_arg(num_localities), this_site_arg(this_locality));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;
        if (this_locality == 0)
        {
            hpx::future<bool> overall_result = reduce_here(reduce_bool_client,
                value, std::logical_or<>{}, generation_arg(i + 1));

            HPX_TEST_EQ(value, overall_result.get());
        }
        else
        {
            hpx::future<void> overall_result = reduce_there(
                reduce_bool_client, std::move(value), generation_arg(i + 1));
            overall_result.get();
        }
    }
}

void test_scatter_bool()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(std::uint32_t(2), num_localities);

    std::uint32_t this_locality = hpx::get_locality_id();

    auto scatter_bool_client =
        hpx::collectives::create_communicator(scatter_bool_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        bool value = i % 2 ? true : false;
        if (this_locality == 0)
        {
            std::vector<bool> data(num_localities);
            for (std::size_t i = 0; i != data.size(); ++i)
            {
                data[i] = value;
            }

            hpx::future<bool> result = scatter_to(
                scatter_bool_client, std::move(data), generation_arg(i + 1));

            HPX_TEST_EQ(value, result.get());
        }
        else
        {
            hpx::future<bool> result =
                scatter_from<bool>(scatter_bool_client, generation_arg(i + 1));

            HPX_TEST_EQ(value, result.get());
        }
    }
}

int hpx_main()
{
    test_all_reduce_bool();
    test_broadcast_bool();
    test_exclusive_scan_bool();
    test_inclusive_scan_bool();
    test_reduce_bool();
    test_scatter_bool();

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
