//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// clang-format off
#include <hpx/config.hpp>

#if defined(HPX_HAVE_HIP) && defined(HPX_HAVE_GPUBLAS)

    #include <hipblas.h>

    #define cublasCreate hipblasCreate
    #define cublasDestroy hipblasDestroy
    #define cublasHandle_t hipblasHandle_t
    #define cublasPointerMode_t hipblasPointerMode_t
    #define cublasSetPointerMode hipblasSetPointerMode
    #define cublasSetStream hipblasSetStream
    #define cublasSgemm hipblasSgemm
    #define cublasStatus_t hipblasStatus_t

    #define CUBLAS_OP_N HIPBLAS_OP_N
    #define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
    #define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
    #define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
    #define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
    #define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
    #define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
    #define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
    #define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
    #define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
    #define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#elif defined(HPX_HAVE_CUDA) && defined(HPX_HAVE_GPUBLAS)

    #include <cublas_v2.h>

#endif
// clang-format on
