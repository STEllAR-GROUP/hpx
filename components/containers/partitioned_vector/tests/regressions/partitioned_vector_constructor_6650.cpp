//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/modules/runtime_distributed.hpp>

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // create as many partitions as we have localities
    [[maybe_unused]] hpx::partitioned_vector<int> data(
        hpx::container_layout(hpx::find_all_localities()));

    return 0;
}

#else

int main()
{
    return 0;
}

#endif
