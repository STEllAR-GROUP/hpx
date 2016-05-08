///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_TARGET_HPP
#define HPX_COMPUTE_CUDA_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace cuda
{
    struct target
    {
        struct native_handle_type
        {
            int device;
            cudaStream_t stream;
        };

        // Constructs default target
        target()
        {
            handle_.device = 0;
            cudaSetDevice(handle_.device);
            cudaStreamCreate(&handle_.stream);
        }

        // Constructs target from a given device ID
        explicit target(int device)
        {
            // FIXME: check for error
            handle_.device = device;
            cudaSetDevice(handle_.device);
            cudaStreamCreate(&handle_.stream);
        }

        native_handle_type const& native_handle() const
        {
            return handle_;
        }

    private:
        native_handle_type handle_;
    };
}}}

#endif
#endif
