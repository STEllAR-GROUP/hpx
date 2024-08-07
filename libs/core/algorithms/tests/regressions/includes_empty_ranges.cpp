//  Copyright (c) 2024 Tobias Wukovitsch
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// An empty subsequence is a subsequence of every sequency, thus if the second
// range is empty, includes should always return true, even if the first range
// is empty.

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

void includes_two_empty_ranges_test()
{
    std::vector<std::size_t> a{};
    std::vector<std::size_t> b{};

    auto result = hpx::includes(
        hpx::execution::par, a.begin(), a.end(), b.begin(), b.end());

    auto expected = std::includes(a.begin(), a.end(), b.begin(), b.end());

    HPX_TEST(result == expected);
}

int hpx_main()
{
    includes_two_empty_ranges_test();
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
