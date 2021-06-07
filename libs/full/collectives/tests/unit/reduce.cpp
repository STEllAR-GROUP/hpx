//  Copyright (c) 2019-2021 Hartmut Kaiser
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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

constexpr char const* reduce_direct_basename = "/test/reduce_direct/";

void test_one_shot_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = this_locality;

        if (this_locality == 0)
        {
            hpx::future<std::uint32_t> overall_result =
                reduce_here(reduce_direct_basename, value,
                    std::plus<std::uint32_t>{}, num_sites_arg(num_localities),
                    this_site_arg(this_locality), generation_arg(i));

            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != num_localities; ++j)
            {
                sum += j;
            }
            HPX_TEST_EQ(sum, overall_result.get());
        }
        else
        {
            hpx::future<void> overall_result =
                reduce_there(reduce_direct_basename, std::move(value),
                    this_site_arg(this_locality), generation_arg(i));
            overall_result.get();
        }
    }
}

void test_multiple_use()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    auto reduce_direct_client = create_communicator(reduce_direct_basename,
        num_sites_arg(num_localities), this_site_arg(this_locality));

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = hpx::get_locality_id();

        if (this_locality == 0)
        {
            hpx::future<std::uint32_t> overall_result = reduce_here(
                reduce_direct_client, value, std::plus<std::uint32_t>{});

            std::uint32_t sum = 0;
            for (std::uint32_t j = 0; j != num_localities; ++j)
            {
                sum += j;
            }
            HPX_TEST_EQ(sum, overall_result.get());
        }
        else
        {
            hpx::future<void> overall_result =
                reduce_there(reduce_direct_client, std::move(value));
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
