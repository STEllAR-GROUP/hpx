//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// If the second range is larger than the first range, search should return the
// past-the-end iterator of the second range.

#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

void search_larger_2nd_range_test()
{
    std::vector<std::size_t> a{1, 2, 3};
    std::vector<std::size_t> b{1, 2, 3, 4};

    auto result = hpx::search(
        hpx::execution::par, a.begin(), a.end(), b.begin(), b.end());

    auto expected = std::search(a.begin(), a.end(), b.begin(), b.end());

    HPX_TEST(result == a.end());
    HPX_TEST(result == expected);
}

int hpx_main()
{
    search_larger_2nd_range_test();
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
