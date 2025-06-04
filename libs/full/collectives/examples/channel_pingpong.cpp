//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>

#include <iostream>

int main()
{
    if (hpx::get_num_localities(hpx::launch::sync) < 2)
    {
        std::cout << "This example requires at least two localities."
                  << std::endl;
        return 0;
    }

    using namespace hpx::collectives;

    std::uint32_t const locality_id = hpx::get_locality_id();
    auto const comm = get_world_channel_communicator();

    if (locality_id == 0)
    {
        set(hpx::launch::sync, comm, that_site_arg(1), 42, tag_arg(123));
        int const received =
            get<int>(hpx::launch::sync, comm, that_site_arg(1), tag_arg(123));
        std::cout << "Received: " << received << "\n";
    }
    if (locality_id == 1)
    {
        int const received =
            get<int>(hpx::launch::sync, comm, that_site_arg(0), tag_arg(123));
        set(hpx::launch::sync, comm, that_site_arg(0), received, tag_arg(123));
    }
    return 0;
}

#else

int main()
{
    return 0;
}

#endif
