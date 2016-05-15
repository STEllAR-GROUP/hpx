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
#include <hpx/exception.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/compute/cuda/target.hpp>

#include <cuda_runtime.h>

#include <string>

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
                    std::string("cudaGetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            if(previous_device_ == target_.native_handle().get_device())
            {
                previous_device_ = -1;
                return;
            }

            error = cudaSetDevice(target_.native_handle().get_device());
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "scoped_active_target::scoped_active_target(target const&)",
                    std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(error));
            }
        }

        ~scoped_active_target()
        {
            cudaError_t error = cudaSuccess;
#if defined(HPX_DEBUG)
            int current_device = -1;
            error = cudaGetDevice(&current_device);
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "scoped_active_target::~scoped_active_target()",
                    std::string("cudaGetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            HPX_ASSERT(current_device == target_.native_handle().get_device());
#endif
            if(previous_device_ != -1)
            {
                error = cudaSetDevice(previous_device_);
                if(error != cudaSuccess)
                {
                    HPX_THROW_EXCEPTION(kernel_error,
                        "scoped_active_target::~scoped_active_target()",
                        std::string("cudaSetDevice failed: ") +
                            cudaGetErrorString(error));
                }
            }
        }

        cudaStream_t stream()
        {
            return target_.native_handle().get_stream();
        }

    private:
        int previous_device_;
        target const& target_;
    };
}}}}

#endif
#endif
