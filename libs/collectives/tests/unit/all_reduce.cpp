//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/collectives.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

char const* all_reduce_basename = "/test/all_reduce/";
char const* all_reduce_direct_basename = "/test/all_reduce_direct/";

HPX_REGISTER_ALLREDUCE(std::uint32_t, test_all_reduce);

int hpx_main(int argc, char* argv[])
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    // test functionality based on future<> of local result
    for (int i = 0; i != 10; ++i)
    {
        hpx::future<std::uint32_t> value =
            hpx::make_ready_future(hpx::get_locality_id());

        hpx::future<std::uint32_t> overall_result =
            hpx::all_reduce(all_reduce_basename, std::move(value),
                std::plus<std::uint32_t>{}, num_localities, i);

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }

    // test functionality based on immediate local result value
    for (int i = 0; i != 10; ++i)
    {
        std::uint32_t value = hpx::get_locality_id();

        hpx::future<std::uint32_t> overall_result =
            hpx::all_reduce(all_reduce_direct_basename, value,
                std::plus<std::uint32_t>{}, num_localities, i);

        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum += j;
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);
    return hpx::util::report_errors();
}
