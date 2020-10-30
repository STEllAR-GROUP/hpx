//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

int hpx_main()
{
    // check number of localities
    HPX_TEST_EQ(
        hpx::util::from_string<std::uint32_t>(
            hpx::get_config_entry("hpx.localities", "")),
                hpx::get_num_localities(hpx::launch::sync));
    HPX_TEST_EQ(
        hpx::util::from_string<std::size_t>(
            hpx::get_config_entry("hpx.os_threads", "")),
                hpx::get_os_thread_count());
    HPX_TEST_EQ(hpx::get_config_entry("hpx.runtime_mode", ""),
        std::string("console"));

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
