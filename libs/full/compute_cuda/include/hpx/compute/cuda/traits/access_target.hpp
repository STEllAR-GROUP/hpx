///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_GPU_SUPPORT)
#include <hpx/async_cuda/target.hpp>
#include <hpx/compute/traits/access_target.hpp>

#include <hpx/async_cuda/custom_gpu_api.hpp>

namespace hpx { namespace compute { namespace traits {
    template <>
    struct access_target<hpx::cuda::experimental::target>
    {
        typedef hpx::cuda::experimental::target target_type;

        template <typename T>
        HPX_HOST_DEVICE static T read(
            hpx::cuda::experimental::target const& tgt, T const* t)
        {
#if defined(__CUDA_ARCH__)
            HPX_UNUSED(tgt);
            return *t;
#else
            T tmp;
            cudaMemcpyAsync(&tmp, t, sizeof(T), cudaMemcpyDeviceToHost,
                tgt.native_handle().get_stream());
            tgt.synchronize();
            return tmp;
#endif
        }

        template <typename T>
        HPX_HOST_DEVICE static void write(
            hpx::cuda::experimental::target const& tgt, T* dst, T const* src)
        {
#if defined(__CUDA_ARCH__)
            HPX_UNUSED(tgt);
            *dst = *src;
#else
            cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyHostToDevice,
                tgt.native_handle().get_stream());
#endif
        }
    };
}}}    // namespace hpx::compute::traits

#endif
