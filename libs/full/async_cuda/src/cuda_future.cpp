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
    runtime_registration_wrapper::runtime_registration_wrapper(hpx::runtime* rt)
      : rt_(rt)
      , registered_(false)
    {
        if (nullptr != hpx::get_runtime_ptr())
        {
            return;
        }

        HPX_ASSERT(rt);

        // Register this thread with HPX, this should be done once for
        // each external OS-thread intended to invoke HPX functionality.
        // Calling this function more than once on the same thread will
        // report an error.
        hpx::error_code ec(hpx::lightweight);    // ignore errors
        hpx::register_thread(rt_, "cuda", ec);
        registered_ = true;
    }

    runtime_registration_wrapper::~runtime_registration_wrapper()
    {
        // Unregister the thread from HPX, this should be done once in the end
        // before the external thread exits, if the runtime registration
        // wrapper actually registered the thread (it may not do so if the
        // wrapper is constructed on a HPX worker thread).
        if (registered_)
        {
            hpx::unregister_thread(rt_);
        }
    }

    hpx::future<void> get_future_with_callback(cudaStream_t stream)
    {
        return get_future_with_callback(
            hpx::util::internal_allocator<>{}, stream);
    }

    hpx::future<void> get_future_with_event(cudaStream_t stream)
    {
        return get_future_with_event(hpx::util::internal_allocator<>{}, stream);
    }
}}}}    // namespace hpx::cuda::experimental::detail
