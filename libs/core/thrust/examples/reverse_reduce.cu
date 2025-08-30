//  Copyright (c)      2025 Aditya Sapra
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

#include <hpx/thrust/algorithms.hpp>
#include <hpx/thrust/policy.hpp>

#include <thrust/device_vector.h>

int hpx_main(int, char**)
{
    hpx::thrust::thrust_device_policy device{};
    thrust::device_vector<int> v(10);
    for (int i = 0; i < 10; ++i)
        v[i] = i + 1;

    hpx::reverse(device, v.begin(), v.end());
    int sum = hpx::reduce(device, v.begin(), v.end(), 0);
    (void) sum;
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
