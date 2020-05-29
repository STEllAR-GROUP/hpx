//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/basic_execution/execution.hpp>
#include <hpx/cuda_support/cuda_future.hpp>
#include <hpx/cuda_support/target.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/futures/future.hpp>

// CUDA runtime
#include <cuda_runtime.h>
// CuBLAS
#include <cublas_v2.h>
//
#include <sstream>

namespace hpx { namespace cuda { namespace experimental {

    // -------------------------------------------------------------------------
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
    // Base class for cuda and cublas executors :
    // Allows you to launch kernels on a stream and get
    // futures back when they are ready
    // -------------------------------------------------------------------------
    struct cuda_executor
    {
        using future_type = hpx::future<void>;

        // -------------------------------------------------------------------------
        // construct - create a cuda stream that all tasks invoked by
        // this helper will use
        cuda_executor(std::size_t device = 0)
          : device_(device)
          , stream_(0)
          , target_(device)
        {
            stream_ = target_.native_handle().get_stream();
        }

        cuda_executor(cuda_executor& other) = delete;
        cuda_executor(const cuda_executor& other) = delete;
        cuda_executor operator=(const cuda_executor& other) = delete;

        ~cuda_executor() {}

        // -------------------------------------------------------------------------
        // OneWay Execution
        // -------------------------------------------------------------------------
        template <typename F, typename... Ts>
        inline void post(F&& f, Ts&&... ts)
        {
            apply(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // -------------------------------------------------------------------------
        // TwoWay Execution
        template <typename F, typename... Ts>
        inline decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return async(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        /*
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
            cuda_error(cudaSetDevice(device_));
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
            cuda_error(cudaSetDevice(device_));
            // launch the kernel directly on the GPU
            cuda_error(cudaLaunchKernel(std::forward<Args>(args)..., stream_));
            return get_future();
        }
*/

        // -------------------------------------------------------------------------
        // launch a kernel on our stream and return a future that will become ready
        // when the task completes, this allows integregration of GPU kernels with
        // hpx::futures and the tasking DAG.
        template <typename R, typename... Params, typename... Args>
        hpx::future<void> async(R (*cuda_kernel)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(device_));
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
            cuda_error(cudaSetDevice(device_));
            // insert the stream handle in the arg list and call the cuda function
            detail::async_helper<R, Params...> helper;
            return helper(cuda_kernel, std::forward<Args>(args)..., stream_);
        }
        /*
        // -------------------------------------------------------------------------
        // launch a task on our stream and pass the error code through from
        // cuda back to the caller, otherwise this function mimics the
        // behaviour of apply.
        template <typename Func, typename... Args>
        cudaError_t apply_pass_through(Func&& cuda_function, Args&&... args)
        {
            // make sure we run on the correct device
            cuda_error(cudaSetDevice(device_));
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
*/
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
                          << "with comp"
                             "ute capability "
                          << target.native_handle().processor_family() << "\n";
            }
        }

    protected:
        int device_;
        cudaStream_t stream_;
        hpx::cuda::target target_;
    };

    // -------------------------------------------------------------------------
    // a simple cublas wrapper helper object that can be used to synchronize
    // cublas calls with an hpx future.
    // -------------------------------------------------------------------------
    struct cublas_executor : hpx::cuda::experimental::cuda_executor
    {
        // construct a cublas stream
        cublas_executor(std::size_t device = 0)
          : hpx::cuda::experimental::cuda_executor(device)
        {
            handle_ = 0;
            hpx::cuda::experimental::cublas_error(cublasCreate(&handle_));
        }

        cublas_executor(cublas_executor& other) = delete;
        cublas_executor(const cublas_executor& other) = delete;
        cublas_executor operator=(const cublas_executor& other) = delete;

        ~cublas_executor()
        {
            hpx::cuda::experimental::cublas_error(cublasDestroy(handle_));
        }

        // -------------------------------------------------------------------------
        // TwoWay Execution
        // -------------------------------------------------------------------------
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return cublas_executor::async(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // -------------------------------------------------------------------------
        // OneWay Execution
        // -------------------------------------------------------------------------
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            cublas_executor::apply(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // -------------------------------------------------------------------------
        // launch a cuBlas function and return a future that will become ready
        // when the task completes, this allows integregration of GPU kernels with
        // hpx::futures and the tasking DAG.
        template <typename R, typename... Params, typename... Args>
        hpx::future<typename std::enable_if<
            std::is_same<cublasStatus_t, R>::value, void>::type>
        async(R (*cublas_function)(Params...), Args&&... args)
        {
            // make sue we run on the correct device
            hpx::cuda::experimental::cuda_error(cudaSetDevice(device_));
            // make sure this operation takes place on our stream
            hpx::cuda::experimental::cublas_error(
                cublasSetStream(handle_, stream_));
            // insert the cublas handle in the arg list and call the cublas function
            hpx::cuda::experimental::detail::async_helper<R, Params...> helper;
            helper(cublas_function, handle_, std::forward<Args>(args)...);
            return get_future();
        }

        // -------------------------------------------------------------------------
        // forward a cuda function through to the cuda executor base class
        template <typename R, typename... Params, typename... Args>
        inline hpx::future<typename std::enable_if<
            std::is_same<cudaError_t, R>::value, void>::type>
        async(R (*cuda_function)(Params...), Args&&... args)
        {
            return cuda_executor::async(
                cuda_function, std::forward<Args>(args)...);
        }

        // This is a simple wrapper for any cublas call, pass in the same arguments
        // that you would use for a cublas call except the cublas handle which is omitted
        // as the wrapper will supply that for you
        template <typename R, typename... Params, typename... Args>
        typename std::enable_if<std::is_same<cublasStatus_t, R>::value, R>::type
        apply(R (*cublas_function)(Params...), Args&&... args)
        {
            // make sue we run on the correct device
            hpx::cuda::experimental::cuda_error(cudaSetDevice(device_));
            // make sure this operation takes place on our stream
            hpx::cuda::experimental::cublas_error(
                cublasSetStream(handle_, stream_));
            // insert the cublas handle in the arg list and call the cublas function
            hpx::cuda::experimental::detail::async_helper<R, Params...> helper;
            return helper(
                cublas_function, handle_, std::forward<Args>(args)...);
        }

        // -------------------------------------------------------------------------
        // forward a cuda function through to the cuda executor base class
        template <typename R, typename... Params, typename... Args>
        inline typename std::enable_if<std::is_same<cudaError_t, R>::value,
            R>::type
        apply(R (*cuda_function)(Params...), Args&&... args)
        {
            return cuda_executor::apply(
                cuda_function, std::forward<Args>(args)...);
        }

        // return a copy of the cublas handle
        cublasHandle_t get_handle()
        {
            return handle_;
        }

    private:
        cublasHandle_t handle_;
    };

}}}    // namespace hpx::cuda::experimental

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<hpx::cuda::experimental::cuda_executor>
      : std::true_type
    {
        // we support fire and forget without returning a waitable/future
    };

    template <>
    struct is_two_way_executor<hpx::cuda::experimental::cuda_executor>
      : std::true_type
    {
        // we support returning a waitable/future
    };

    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<hpx::cuda::experimental::cublas_executor>
      : std::true_type
    {
        // we support fire and forget without returning a waitable/future
    };

    template <>
    struct is_two_way_executor<hpx::cuda::experimental::cublas_executor>
      : std::true_type
    {
        // we support returning a waitable/future
    };

    /// \endcond
}}}    // namespace hpx::parallel::execution
