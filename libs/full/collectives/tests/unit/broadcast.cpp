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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

constexpr char const* broadcast_basename = "/test/broadcast/";
constexpr char const* broadcast_direct_basename = "/test/broadcast_direct/";

int hpx_main()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(std::uint32_t(2), num_localities);

    // test functionality based on future<> of local result
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (hpx::get_locality_id() == 0)
        {
            hpx::future<std::uint32_t> result = broadcast_to(broadcast_basename,
                hpx::make_ready_future(i + 42), num_localities, i);

            HPX_TEST_EQ(i + 42, result.get());
        }
        else
        {
            hpx::future<std::uint32_t> result =
                hpx::broadcast_from<std::uint32_t>(broadcast_basename, i);

            HPX_TEST_EQ(i + 42, result.get());
        }
    }

    // test functionality based on immediate local result value
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (hpx::get_locality_id() == 0)
        {
            hpx::future<std::uint32_t> result = hpx::broadcast_to(
                broadcast_direct_basename, i + 42, num_localities, i);

            HPX_TEST_EQ(i + 42, result.get());
        }
        else
        {
            hpx::future<std::uint32_t> result =
                hpx::broadcast_from<std::uint32_t>(
                    broadcast_direct_basename, i);

            HPX_TEST_EQ(i + 42, result.get());
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
