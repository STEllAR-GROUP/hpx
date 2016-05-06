///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_GET_TARGETS_HPP
#define HPX_COMPUTE_CUDA_GET_TARGETS_HPP

#include <hpx/config.hpp>

#include <cuda_runtime.h>

#include <vector>

namespace hpx { namespace compute { namespace cuda
{
    std::vector<target> get_targets()
    {
        int device_count = 0;
        // FIXME: check for error
        cudaGetDeviceCount(&device_count);
        std::vector<target> targets;
        targets.reserve(device_count);

        for(int i = 0; i < device_count; ++i)
        {
            targets.push_back(target(i));
        }

        return targets;
    }
}}}

#endif
