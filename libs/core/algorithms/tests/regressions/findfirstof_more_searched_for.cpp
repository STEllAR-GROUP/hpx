//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// It should be no issue for find_first_of if the range being searched
// is smaller than the range of the searched elements.

#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

void find_first_of_failing_test()
{
    std::vector<std::size_t> a{1, 2, 3, 4, 5};
    std::vector<std::size_t> b{10, 11, 12, 2, 13, 14, 15};

    auto result = hpx::find_first_of(
        hpx::execution::par, a.begin(), a.end(), b.begin(), b.end());
    auto expected = ++a.begin();

    HPX_TEST(result == expected);
}

int hpx_main()
{
    find_first_of_failing_test();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
