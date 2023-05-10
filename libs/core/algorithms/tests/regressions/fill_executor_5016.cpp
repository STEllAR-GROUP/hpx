//  Copyright (c) 2020 Steven R. Brandt
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #5016: hpx::parallel::fill fails compiling

// suppress deprecation warnings for algorithms
#define HPX_HAVE_DEPRECATION_WARNINGS_V1_6 0

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <vector>

void fill_example()
{
    hpx::execution::parallel_executor exec;

    std::vector<float> vd(5);
    hpx::fill(hpx::execution::par.on(exec), vd.begin(), vd.end(), 2.0f);

    std::vector<float> vd1(5);
    hpx::fill(hpx::execution::par.on(exec), vd1.begin(), vd1.end(), 2.0f);

    std::vector<float> expected(5);
    std::fill(expected.begin(), expected.end(), 2.0f);

    HPX_TEST(vd == expected);
    HPX_TEST(vd1 == expected);
}

int hpx_main()
{
    fill_example();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
