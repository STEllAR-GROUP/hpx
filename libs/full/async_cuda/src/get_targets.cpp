//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/async_cuda/target.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/vector.hpp>
#endif

#include <hpx/async_cuda/target.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <hpx/async_cuda/custom_gpu_api.hpp>

namespace hpx { namespace cuda { namespace experimental {
    std::vector<target> get_local_targets()
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::experimental::get_local_targets()",
                std::string("cudaGetDeviceCount failed: ") +
                    cudaGetErrorString(error));
        }

        std::vector<target> targets;
        targets.reserve(device_count);

        for (int i = 0; i < device_count; ++i)
        {
            targets.emplace_back(target(i));
        }

        return targets;
    }

    void print_local_targets(void)
    {
        auto targets = get_local_targets();
        for (auto target : targets)
        {
            std::cout << "GPU Device " << target.native_handle().get_device()
                      << ": \"" << target.native_handle().processor_name()
                      << "\" "
                      << "with compute capability "
                      << target.native_handle().processor_family() << "\n";
        }
    }

}}}    // namespace hpx::cuda::experimental

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
HPX_PLAIN_ACTION(
    hpx::cuda::experimental::get_local_targets, cuda_get_targets_action);

namespace hpx { namespace cuda { namespace experimental {
    hpx::future<std::vector<target>> get_targets(hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
            return hpx::make_ready_future(get_local_targets());

        return hpx::async(cuda_get_targets_action(), locality);
    }
}}}    // namespace hpx::cuda::experimental
#endif
