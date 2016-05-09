///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_DETAIL_SCOPED_ACTIVE_TARGET_HPP
#define HPX_COMPUTE_CUDA_DETAIL_SCOPED_ACTIVE_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/compute/cuda/target.hpp>
#include <hpx/util/assert.hpp>

#include <cuda_runtime.h>

namespace hpx { namespace compute { namespace cuda { namespace detail
{
    struct scoped_active_target
    {
        scoped_active_target(target const& t)
          : previous_device_(-1)
          , target_(t)
        {
            cudaError_t error = cudaGetDevice(&previous_device_);
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "scoped_active_target::scoped_active_target(target const&)",
                    "cudaGetDevice failed");
            }
            if(previous_device_ == target_.native_handle().device_)
            {
                previous_device_ = -1;
                return;
            }

            error = cudaSetDevice(target_.native_handle().device_);
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "scoped_active_target::scoped_active_target(target const&)",
                    "cudaSetDevice failed");
            }
        }

        ~scoped_active_target()
        {
#if defined(HPX_DEBUG)
            int current_device = -1;
            cudaError_t error = cudaGetDevice(&current_device);
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "scoped_active_target::~scoped_active_target()",
                    "cudaGetDevice failed");
            }
            HPX_ASSERT(current_device == target_.native_handle().device_);
#endif
            if(previous_device_ != -1)
            {
                error = cudaSetDevice(previous_device_);
                if(error != cudaSuccess)
                {
                    HPX_THROW_EXCEPTION(kernel_error,
                        "scoped_active_target::~scoped_active_target()",
                        "cudaSetDevice failed");
                }
            }
        }

        cudaStream_t stream()
        {
            return target_.native_handle().stream_;
        }

    private:
        int previous_device_;
        target const& target_;
    };
}}}}

#endif
#endif
