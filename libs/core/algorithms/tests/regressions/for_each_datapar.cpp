//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

template <typename ExPolicy>
void test_for_each(ExPolicy&& policy)
{
    std::vector<int> c(1027);
    // this test ensures that aligned is working properly for various inputs
    for (size_t first = 0; first < c.size(); ++first)
        hpx::for_each(policy,
            std::begin(c) + static_cast<std::ptrdiff_t>(first), std::end(c),
            [](auto& val) { ++val; });

    std::size_t count = 0;
    int ans = 0;
    std::for_each(std::begin(c), std::end(c), [&count, &ans](int v) -> void {
        HPX_TEST_EQ(v, ++ans);
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

void for_each_test()
{
    using namespace hpx::execution;
    test_for_each(simd);
    test_for_each(par_simd);
}

int hpx_main()
{
    for_each_test();
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
