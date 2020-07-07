//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/modules/testing.hpp>
#include "iter_sent.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>

void myfunction(int i)
{
    std::cout << ' ' << i;
}

int main()
{
    hpx::parallel::for_each(hpx::parallel::execution::seq,
        Iterator<std::int64_t>{0}, Sentinel<int64_t>{100}, &myfunction);

    //HPX_TEST_EQ(result, std::int64_t(4950));

    return hpx::util::report_errors();
}
