//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)

#include <hpx/exception.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <hpx/compute/cuda/target.hpp>

#include <string>

#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace cuda
{
    std::vector<target> get_local_targets();
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::get_local_targets()",
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

HPX_PLAIN_ACTION(hpx::compute::host::get_local_targets,
    compute_cuda_get_targets_action);

namespace hpx { namespace compute { namespace cuda
{
    hpx::future<std::vector<target> > get_targets(hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
            return hpx::make_ready_future(get_local_targets());

        return hpx::async(compute_cuda_get_targets_action(), locality);
    }
}}}

#endif

