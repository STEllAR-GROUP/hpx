///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_DETAIL_SCOPED_ACTIVE_TARGET_HPP
#define HPX_COMPUTE_CUDA_DETAIL_SCOPED_ACTIVE_TARGET_HPP

#include <hpx/config.hpp>
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
            // FIXME: check for error
            cudaGetDevice(&previous_device_);
            if(previous_device_ == target_.native_handle().device)
            {
                previous_device_ = -1;
                return;
            }

            cudaSetDevice(target_.native_handle().device);
        }

        ~scoped_active_target()
        {
#if defined(HPX_DEBUG)
            int current_device = -1;
            cudaGetDevice(&current_device);
            HPX_ASSERT(current_device == target_.native_handle().device);
#endif
            if(previous_device_ != -1)
            {
                // FIXME: check for error
                cudaSetDevice(previous_device_);
            }
        }

        cudaStream_t stream()
        {
            return target_.native_handle().stream;
        }

    private:
        int previous_device_;
        target const& target_;
    };
}}}}

#endif
