///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_GET_TARGETS_HPP
#define HPX_COMPUTE_CUDA_GET_TARGETS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/exception.hpp>
#include <hpx/compute/cuda/target.hpp>

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace hpx { namespace compute { namespace cuda
{
    std::vector<target> get_targets()
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::get_targets()",
                std::string("cudaGetDeviceCount failed: ") +
                    cudaGetErrorString(error));
        }

        std::vector<target> targets;
        targets.reserve(device_count);

        for(int i = 0; i < device_count; ++i)
        {
            targets.emplace_back(target(i));
        }

        return targets;
    }
}}}

#endif
#endif
