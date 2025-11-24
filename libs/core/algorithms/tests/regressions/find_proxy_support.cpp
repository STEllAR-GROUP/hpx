//  Copyright (c) 2025 Arivoli Ramamoorthy
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Regression test for Issue #6719
//  Ensures that the find-family algorithms correctly handle proxy-reference

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

void find_proxy_reference_test()
{
    std::vector<bool> v = {false, true, false, true};

    // Sequential policy
    auto it_seq = hpx::find(hpx::execution::seq, v.begin(), v.end(), true);
    HPX_TEST(it_seq != v.end());
    HPX_TEST_EQ(std::distance(v.begin(), it_seq), 1);

    // Parallel policy
    auto it_par = hpx::find(hpx::execution::par, v.begin(), v.end(), false);
    HPX_TEST(it_par != v.end());
    HPX_TEST_EQ(std::distance(v.begin(), it_par), 0);

    // find_if
    auto it_if = hpx::find_if(
        hpx::execution::seq, v.begin(), v.end(), [](bool x) { return x; });
    HPX_TEST(it_if != v.end());
    HPX_TEST_EQ(std::distance(v.begin(), it_if), 1);

    // find_if_not
    auto it_not = hpx::find_if_not(
        hpx::execution::par, v.begin(), v.end(), [](bool x) { return x; });
    HPX_TEST(it_not != v.end());
    HPX_TEST_EQ(std::distance(v.begin(), it_not), 0);
}

int hpx_main()
{
    find_proxy_reference_test();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
