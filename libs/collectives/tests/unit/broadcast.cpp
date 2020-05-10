//  Copyright (c) 2020 Hartmut Kaiser
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

char const* broadcast_basename = "/test/broadcast/";
char const* broadcast_direct_basename = "/test/broadcast_direct/";

HPX_REGISTER_BROADCAST(std::uint32_t, test_broadcast);

int hpx_main(int argc, char* argv[])
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST(num_localities >= 2);

    // test functionality based on future<> of local result
    for (std::uint32_t i = 0; i != 10; ++i)
    {
        if (hpx::get_locality_id() == 0)
        {
            hpx::future<void> f = broadcast_to(broadcast_basename,
                hpx::make_ready_future(i + 42), num_localities, i);
            f.get();
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
            hpx::future<void> f = hpx::broadcast_to(
                broadcast_direct_basename, i + 42, num_localities, i);
            f.get();
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

    HPX_TEST_EQ(hpx::init(argc, argv, cfg), 0);
    return hpx::util::report_errors();
}
