//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
//
#include <hpx/cuda_support/cuda_future_helper.hpp>

// CUDA runtime
#include <cuda_runtime.h>
// CuBLAS
#include <cublas_v2.h>
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace hpx { namespace cuda {

    // -------------------------------------------------------------------------
    // a simple cublas wrapper helper object that can be used to synchronize
    // cublas calls with an hpx future.
    // -------------------------------------------------------------------------
    template <typename T>
    struct cublas_helper : hpx::cuda::cuda_future_helper
    {
        // construct a cublas stream
        cublas_helper(std::size_t device = 0)
          : hpx::cuda::cuda_future_helper(device)
        {
            handle_ = 0;
            hpx::cuda::cublas_error(cublasCreate(&handle_));
        }

        cublas_helper(cublas_helper& other) = delete;
        cublas_helper(const cublas_helper& other) = delete;
        cublas_helper operator=(const cublas_helper& other) = delete;

        ~cublas_helper()
        {
            hpx::cuda::cublas_error(cublasDestroy(handle_));
        }

        // -------------------------------------------------------------------------
        // launch a cuBlas function and return a future that will become ready
        // when the task completes, this allows integregration of GPU kernels with
        // hpx::futuresa and the tasking DAG.
        template <typename R, typename... Params, typename... Args>
        hpx::future<void> async(R (*cublas_function)(Params...), Args&&... args)
        {
            // make sue we run on the correct device
            hpx::cuda::cuda_error(
                cudaSetDevice(target_.native_handle().get_device()));
            // make sure this operation takes place on our stream
            hpx::cuda::cublas_error(cublasSetStream(handle_, stream_));
            // insert the cublas handle in the arg list and call the cublas function
            hpx::cuda::detail::async_helper<R, Params...> helper;
            helper(cublas_function, handle_, std::forward<Args>(args)...);
            return get_future();
        }

        // This is a simple wrapper for any cublas call, pass in the same arguments
        // that you would use for a cublas call except the cublas handle which is omitted
        // as the wrapper will supply that for you
        template <typename R, typename... Params, typename... Args>
        R apply(R (*cublas_function)(Params...), Args&&... args)
        {
            // make sue we run on the correct device
            hpx::cuda::cuda_error(
                cudaSetDevice(target_.native_handle().get_device()));
            // make sure this operation takes place on our stream
            hpx::cuda::cublas_error(cublasSetStream(handle_, stream_));
            // insert the cublas handle in the arg list and call the cublas function
            hpx::cuda::detail::async_helper<R, Params...> helper;
            return helper(
                cublas_function, handle_, std::forward<Args>(args)...);
        }

        // return a copy of the cublas handle
        cublasHandle_t get_handle()
        {
            return handle_;
        }

    private:
        cublasHandle_t handle_;
    };

}}    // namespace hpx::cuda

//#endif
