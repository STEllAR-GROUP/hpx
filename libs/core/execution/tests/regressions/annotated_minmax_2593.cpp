//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

double compute_minmax(const std::vector<double> v)
{
    hpx::execution::experimental::static_chunk_size param;
    hpx::execution::parallel_task_policy par_policy;
    auto policy = par_policy.with(param);

    auto minmaxX_ = hpx::minmax_element(policy, v.begin(), v.end());
    auto minmaxX = minmaxX_.get();
    return *minmaxX.max - *minmaxX.min;
}

int hpx_main()
{
    std::vector<double> vec = {1.2, 3.4, 2.3, 77.8};
    double extent;

    hpx::async(hpx::launch::sync,
        hpx::annotated_function(
            [&]() { extent = compute_minmax(vec); }, "compute_minmax"));
    HPX_TEST_EQ(extent, 76.6);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
