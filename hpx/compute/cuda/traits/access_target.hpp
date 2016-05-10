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
#include <hpx/compute/traits/access_target.hpp>
#include <hpx/compute/cuda/target.hpp>

#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace traits
{
    template <>
    struct access_target<cuda::target>
    {
        typedef cuda::target target_type;

        template <typename T>
        HPX_HOST_DEVICE
        static T& access(cuda::target const& tgt, T * t, std::size_t pos)
        {
#if defined(__CUDA_ARCH__)
            return *(t + pos);
#else
            static T tmp;
            cudaMemcpyAsync(&tmp, t + pos, sizeof(T), cudaMemcpyDeviceToHost,
                tgt.native_handle().stream_);
            return tmp;
#endif
        }
    };
}}}

#endif
#endif
