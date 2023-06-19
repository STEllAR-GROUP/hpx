//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/async_cuda/custom_gpu_api.hpp>

namespace hpx { namespace cuda { namespace experimental { namespace detail {
    hpx::future<void> get_future_with_callback(cudaStream_t stream)
    {
        return get_future_with_callback(
            hpx::util::internal_allocator<>{}, stream);
    }

    hpx::future<void> get_future_with_event(cudaStream_t stream, int device)
    {
        return get_future_with_event(
            hpx::util::internal_allocator<>{}, stream, device);
    }
}}}}    // namespace hpx::cuda::experimental::detail
