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

#include <hpx/async_cuda/cuda_polling_helper.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <thrust/device_vector.h>

int hpx_main(int, char**)
{
    // Enable CUDA event polling on the "default" pool for HPX <-> CUDA future bridging
    hpx::cuda::experimental::enable_user_polling polling_guard("default");

    hpx::thrust::thrust_task_policy task{};
    thrust::device_vector<int> v(1024, 0);
    auto f = hpx::fill_n(task, v.begin(), 512, 9);
    f.get();
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init_params init_args;
    return hpx::local::init(hpx_main, argc, argv, init_args);
}
