//  Copyright (c) 2019 Austin McCartney
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3641: Trouble with using ranges-v3 and hpx::parallel::reduce
// #3646: Parallel algorithms should accept iterator/sentinel pairs

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/modules/testing.hpp>
#include "iter_sent.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>

int main()
{
    std::int64_t result = hpx::ranges::reduce(hpx::parallel::execution::seq,
        Iterator<std::int64_t>{0}, Sentinel<int64_t>{100}, std::int64_t(0));

    HPX_TEST_EQ(result, std::int64_t(4950));

    result = hpx::ranges::reduce(hpx::parallel::execution::par,
        Iterator<std::int64_t>{0}, Sentinel<int64_t>{100}, std::int64_t(0));

    HPX_TEST_EQ(result, std::int64_t(4950));

    return hpx::util::report_errors();
}
