//  Copyright (c) 2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

// #define BOOST_NO_CXX11_ALLOCATOR
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
//
#include <hpx/cuda_support/target.hpp>

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

// -------------------------------------------------------------------------
// A simple cuda wrapper helper object that can be used to synchronize
// calls with an hpx future.
//
// NOTE: The implementation in this file is intended to be compiled with the
// normal C++ compiler and so we cannot directly call a cuda kernel using the
// <<<>>> operator. We therefore require the kernel to be compiled in another
// file by nvcc/cuda-clang and a wrapper function provided that has c++ linkage
//
//        Here is a trivial kernel that can be invoked on the GPU
//        __global__ int trivial_kernel(double val) {
//          printf("hello from gpu with value %f\n", val);
//          return 0;
//        }
//
//        Here is a trivial wrapper that must be compiled into the *.cu
//        int call_trivial_kernel(double val) {
//            return trivial_kernel(val);
//        }
//
// From the user C++ code we may now declare the wrapper and call it with our helper
//        extern int call_trivial_kernel(double val);
//        auto fut = helper.async(&call_trivial_kernel, 3.1415);
//
// -------------------------------------------------------------------------
namespace hpx { namespace cuda {

    // -------------------------------------------------------------------------
    // Forward declare these error utility functions
    void cuda_error(cudaError_t err);
    void cublas_error(cublasStatus_t err);

    namespace detail {

        // -------------------------------------------------------------------------
        // Error handling in cublas calls
        // not all of these are supported by all cuda/cublas versions
        // (comment them out if they cause compiler errors)
        inline const char* _cublasGetErrorEnum(cublasStatus_t error)
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
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            }
            return "<unknown>";
        }

        // -------------------------------------------------------------------------
        // To save writing nearly the same code N times, we will use a helper
        // that we can specialize for cuda Error types
        template <typename R, typename... Args>
        struct async_helper;

        // default implementation
        template <typename R, typename... Args>
        struct async_helper
        {
            inline R operator()(R (*f)(Args...), Args... args)
            {
                return f(args...);
            }
        };

        // specialize invoker helper for return type void
        template <typename... Args>
        struct async_helper<void, Args...>
        {
            inline void operator()(void (*f)(Args...), Args... args)
            {
                f(args...);
            }
        };

        // specialize invoker helper for return type of cudaError_t
        template <typename... Args>
        struct async_helper<cudaError_t, Args...>
        {
            inline cudaError_t operator()(
                cudaError_t (*f)(Args...), Args... args)
            {
                cudaError_t err = f(args...);
                cuda_error(err);
                return err;
            }
        };

        // specialize invoker helper for return type of cublasStatus_t
        template <typename... Args>
        struct async_helper<cublasStatus_t, Args...>
        {
            inline cublasStatus_t operator()(
                cublasStatus_t (*f)(Args...), Args... args)
            {
                cublasStatus_t err = f(args...);
                cublas_error(err);
                return err;
            }
        };
    }    // namespace detail

    // -------------------------------------------------------------------------
    // Error message handling for cuda and cublas
    // -------------------------------------------------------------------------
    inline void cuda_error(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::stringstream temp;
            temp << "cuda function returned error code "
                 << cudaGetErrorString(err);
            throw std::runtime_error(temp.str());
        }
    }

    inline void cublas_error(cublasStatus_t err)
    {
        if (err != CUBLAS_STATUS_SUCCESS)
        {
            std::stringstream temp;
            temp << "cublas function returned error code "
                 << detail::_cublasGetErrorEnum(err);
            throw std::runtime_error(temp.str());
        }
    }

    // -------------------------------------------------------------------------
    // Cuda future helper
    // Allows you to launch kernels on a stream and get
    // futures back when they are ready
    // -------------------------------------------------------------------------
    struct cuda_future_helper
    {
        using future_type = hpx::future<void>;

        // construct - create a cuda stream that all tasks invoked by
        // this helper will use
        cuda_future_helper(std::size_t device = 0)
          : target_(device)
        {
            stream_ = target_.native_handle().get_stream();
        }

        ~cuda_future_helper() {}

        // -------------------------------------------------------------------------
        // launch a kernel on our stream - this does not require a c++ wrapped
        // invoke call of the cuda kernel but must be called with the args that would
        // otherwise be passed to cudaLaunchKernel - minus the stream arg which
        // the helper class will provide. This function does not return a future.
        // Typically, one must pass ...
        // const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem)
        template <typename R, typename... Params, typename... Args>
        R device_launch_apply(R (*cuda_kernel)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            // launch the kernel directly on the GPU
            cuda_error(
                cudaLaunchKernel(reinterpret_cast<void const*>(cuda_kernel),
                    std::forward<Args>(args)..., stream_));
        }

        // -------------------------------------------------------------------------
        // launch a kernel on our stream - this does not require a c++ wrapped
        // invoke call of the cuda kernel but must be called with the args that would
        // otherwise be passed to cudaLaunchKernel - minus the stream arg which
        // the helper class will provide.
        // This function returns a future that will become ready when the task
        // completes, this allows integregration of GPU kernels with
        // hpx::futures and the tasking DAG.
        //
        // Typically, for cudaLaunchKernel one must pass ...
        // const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem)
        template <typename... Args>
        hpx::future<void> device_launch_async(Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            // launch the kernel directly on the GPU
            cuda_error(cudaLaunchKernel(std::forward<Args>(args)..., stream_));
            return get_future();
        }

        // -------------------------------------------------------------------------
        // launch a kernel on our stream and return a future that will become ready
        // when the task completes, this allows integregration of GPU kernels with
        // hpx::futures and the tasking DAG.
        template <typename R, typename... Params, typename... Args>
        hpx::future<void> async(R (*cuda_kernel)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            // insert the stream handle in the arg list and call the cuda function
            detail::async_helper<R, Params...> helper;
            helper(cuda_kernel, std::forward<Args>(args)..., stream_);
            return get_future();
        }

        // -------------------------------------------------------------------------
        // launch a kernel on our stream and return without a future
        template <typename R, typename... Params, typename... Args>
        R apply(R (*cuda_kernel)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            // insert the stream handle in the arg list and call the cuda function
            detail::async_helper<R, Params...> helper;
            return helper(cuda_kernel, std::forward<Args>(args)..., stream_);
        }

        // -------------------------------------------------------------------------
        // launch a task on our stream and pass the error code through from
        // cuda back to the caller, otherwise this function mimics the
        // behaviour of apply.
        template <typename Func, typename... Args>
        cudaError_t apply_pass_through(Func&& cuda_function, Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(target_.native_handle().get_device()));
            // insert the stream handle in the arg list and call the cuda function
            return cuda_function(std::forward<Args>(args)..., stream_);
        }

        // -------------------------------------------------------------------------
        // utility function for memory copies to/from the GPU, async and apply versions
        template <typename... Args>
        hpx::future<void> memcpy_async(Args&&... args)
        {
            return async(cudaMemcpyAsync, std::forward<Args>(args)...);
        }

        template <typename... Args>
        auto memcpy_apply(Args&&... args)
        {
            return apply(cudaMemcpyAsync, std::forward<Args>(args)...);
        }

        // -------------------------------------------------------------------------
        // utility function for setting memory on the GPU, async and apply versions
        template <typename... Args>
        hpx::future<void> memset_async(Args&&... args)
        {
            return async(cudaMemsetAsync, std::forward<Args>(args)...);
        }

        template <typename... Args>
        auto memset_apply(Args&&... args)
        {
            return apply(cudaMemsetAsync, std::forward<Args>(args)...);
        }

        // -------------------------------------------------------------------------
        // get the future to synchronize this cublas stream with
        future_type get_future()
        {
            return target_.get_future();
        }

        // -------------------------------------------------------------------------
        // return a reference to the compute::cuda object owned by this class
        hpx::cuda::target& get_target()
        {
            return target_;
        }

        // -------------------------------------------------------------------------
        cudaStream_t get_stream()
        {
            return stream_;
        }

        // -------------------------------------------------------------------------
        // utility function to print target information for this helper object
        static void print_local_targets(void)
        {
            auto targets = hpx::cuda::target::get_local_targets();
            for (auto target : targets)
            {
                std::cout << "GPU Device "
                          << target.native_handle().get_device() << ": \""
                          << target.native_handle().processor_name() << "\" "
                          << "with compute capability "
                          << target.native_handle().processor_family() << "\n";
            }
        }

    protected:
        cudaStream_t stream_;
        hpx::cuda::target target_;
    };

}}    // namespace hpx::cuda

//#endif
