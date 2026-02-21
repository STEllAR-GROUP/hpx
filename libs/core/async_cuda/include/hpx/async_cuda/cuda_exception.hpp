//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>

// CUDA runtime
#include <hpx/async_cuda/custom_gpu_api.hpp>

#include <string>

namespace hpx::cuda::experimental {

    // -------------------------------------------------------------------------
    // exception type for failed launch of cuda functions
    HPX_CXX_CORE_EXPORT struct HPX_ALWAYS_EXPORT cuda_exception : hpx::exception
    {
        cuda_exception(std::string const& msg, cudaError_t const err)
          : hpx::exception(hpx::error::bad_function_call, msg)
          , err_(err)
        {
        }

        cudaError_t get_cuda_errorcode() const
        {
            return err_;
        }

    protected:
        cudaError_t err_;
    };

    // -------------------------------------------------------------------------
    // Error message handler for cuda calls
    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT void check_cuda_error(
        cudaError_t const err);
}    // namespace hpx::cuda::experimental
