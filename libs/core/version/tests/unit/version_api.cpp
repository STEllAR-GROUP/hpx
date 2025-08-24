//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/modules/version.hpp>

#include <cstdint>

int main()
{
    HPX_TEST_EQ(
        hpx::major_version(), static_cast<std::uint8_t>(HPX_VERSION_MAJOR));
    HPX_TEST_EQ(
        hpx::minor_version(), static_cast<std::uint8_t>(HPX_VERSION_MINOR));
    HPX_TEST_EQ(hpx::subminor_version(),
        static_cast<std::uint8_t>(HPX_VERSION_SUBMINOR));
    HPX_TEST_EQ(
        hpx::full_version(), static_cast<std::uint32_t>(HPX_VERSION_FULL));

    return 0;
}
