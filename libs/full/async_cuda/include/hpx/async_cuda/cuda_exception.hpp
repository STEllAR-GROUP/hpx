//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception.hpp>

// CUDA runtime
#include <cuda_runtime.h>
//
#include <string>

namespace hpx { namespace cuda { namespace experimental {

    // -------------------------------------------------------------------------
    // exception type for failed launch of cuda functions
    struct HPX_EXPORT cuda_exception : hpx::exception
    {
        cuda_exception(const std::string& msg, cudaError_t err)
          : hpx::exception(hpx::bad_function_call, msg)
          , err_(err)
        {
        }
        cudaError_t get_cuda_errorcode()
        {
            return err_;
        }

    protected:
        cudaError_t err_;
    };

    // -------------------------------------------------------------------------
    // Error message handler for cuda calls
    inline cudaError_t check_cuda_error(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            auto temp = std::string("cuda function returned error code :") +
                cudaGetErrorString(err);
            throw cuda_exception(temp, err);
        }
        return err;
    }
}}}    // namespace hpx::cuda::experimental
