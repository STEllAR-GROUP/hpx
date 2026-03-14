//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//  Copyright (c) 2024-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_GPU_SUPPORT) && defined(HPX_HAVE_GPUBLAS)
#include <hpx/async_cuda/cublas_executor.hpp>

#include <string>

namespace hpx::cuda::experimental {

    namespace detail {
        // --------------------------------------------------------------------
        // Error handling in cublas calls
        // not all of these are supported by all cuda/cublas versions
        // (comment them out if they cause compiler errors)
        char const* _cublasGetErrorString(cublasStatus_t error)
        {
            switch (error)
            {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
#ifdef HPX_HAVE_HIP
            case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
                return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
#if HPX_HIP_VERSION >= 40300000
            case HIPBLAS_STATUS_INVALID_ENUM:
                return "HIPBLAS_STATUS_INVALID_ENUM";
#endif
            case HIPBLAS_STATUS_UNKNOWN:
                return "HIPBLAS_STATUS_UNKNOWN";
#else
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
            default:
                break;
            }
            return "<unknown>";
        }
    }    // namespace detail

    cublasStatus_t check_cublas_error(cublasStatus_t err)
    {
        if (err != CUBLAS_STATUS_SUCCESS)
        {
            auto temp = std::string("cublas function returned error code :") +
                detail::_cublasGetErrorString(err);
            throw cublas_exception(temp, err);
        }
        return err;
    }
}    // namespace hpx::cuda::experimental

#endif
