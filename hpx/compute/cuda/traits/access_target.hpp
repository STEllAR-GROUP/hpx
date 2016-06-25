///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_TRAITS_HPP
#define HPX_COMPUTE_CUDA_TARGET_TRAITS_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/traits/access_target.hpp>

#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace traits
{
    template <>
    struct access_target<cuda::target>
    {
        typedef cuda::target target_type;

        template <typename T>
        HPX_HOST_DEVICE
        static T read(cuda::target const& tgt, T const* t)
        {
#if defined(__CUDA_ARCH__)
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
        HPX_HOST_DEVICE
        static void write(cuda::target const& tgt, T* dst, T const* src)
        {
#if defined(__CUDA_ARCH__)
            *dst = *src;
#else
            cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyHostToDevice,
                tgt.native_handle().get_stream());
#endif
        }
    };
}}}

#endif
#endif
