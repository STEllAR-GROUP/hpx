//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/cuda_executor.hpp>
#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/futures/future.hpp>

// CUDA runtime
#include <cuda_runtime.h>
// CuBLAS
#include <cublas_v2.h>
//
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace cuda { namespace experimental {

    namespace detail {
        using print_on = debug::enable_print<false>;
        static constexpr print_on cub_debug("CUBLAS_");

        // -------------------------------------------------------------------------
        // Error handling in cublas calls
        // not all of these are supported by all cuda/cublas versions
        // (comment them out if they cause compiler errors)
        inline const char* _cublasGetErrorString(cublasStatus_t error)
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
    }    // namespace detail

    // -------------------------------------------------------------------------
    // exception type for failed launch of cuda functions
    struct HPX_EXPORT cublas_exception : hpx::exception
    {
        cublas_exception(const std::string& msg, cublasStatus_t err)
          : hpx::exception(hpx::bad_function_call, msg)
          , err_(err)
        {
        }
        cublasStatus_t get_cublas_errorcode()
        {
            return err_;
        }

    protected:
        cublasStatus_t err_;
    };

    inline cublasStatus_t check_cublas_error(cublasStatus_t err)
    {
        if (err != CUBLAS_STATUS_SUCCESS)
        {
            auto temp = std::string("cublas function returned error code :") +
                detail::_cublasGetErrorString(err);
            throw cublas_exception(temp, err);
        }
        return err;
    }

    namespace detail {
        // specialization for return type of cublasStatus_t
        template <typename... Args>
        struct dispatch_helper<cublasStatus_t, Args...>
        {
            inline cublasStatus_t operator()(
                cublasStatus_t (*f)(Args...), Args... args)
            {
                cublasStatus_t err = f(args...);
                return check_cublas_error(err);
            }
        };

        struct cublas_handle
        {
            static cublasHandle_t create()
            {
                cublasHandle_t handle;
                check_cublas_error(cublasCreate(&handle));
                return handle;
            }

            // deleter for shared_ptr
            void operator()(cublasHandle_t handle) const
            {
                check_cublas_error(cublasDestroy(handle));
            }
        };
    }    // namespace detail

    // -------------------------------------------------------------------------
    // a simple cublas wrapper helper object that can be used to synchronize
    // cublas calls with an hpx future.
    // -------------------------------------------------------------------------
    struct cublas_executor : cuda_executor
    {
        // cublas handle is type : struct cublasContext *
        using handle_ptr = std::shared_ptr<struct cublasContext>;

        // construct a cublas stream
        cublas_executor(std::size_t device,
            cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST,
            bool event_mode = false)
          : hpx::cuda::experimental::cuda_executor(device, event_mode)
          , pointer_mode_(pointer_mode)
        {
            detail::cub_debug.debug(
                debug::str<>("cublas_executor"), "event mode", event_mode);

            handle_ = handle_ptr(
                detail::cublas_handle::create(), detail::cublas_handle{});
        }

        ~cublas_executor() {}

        // -------------------------------------------------------------------------
        // OneWay Execution
        // -------------------------------------------------------------------------
        template <typename F, typename... Ts>
        decltype(auto) post(F&& f, Ts&&... ts)
        {
            return cublas_executor::apply(
                std::forward<F>(f), std::forward<Ts>(ts)...);
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

    protected:
        // This is a simple wrapper for any cublas call, pass in the same arguments
        // that you would use for a cublas call except the cublas handle which is omitted
        // as the wrapper will supply that for you
        template <typename R, typename... Params, typename... Args>
        typename std::enable_if<std::is_same<cublasStatus_t, R>::value, R>::type
        apply(R (*cublas_function)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            check_cuda_error(cudaSetDevice(device_));
            // make sure this operation takes place on our stream
            check_cublas_error(cublasSetStream(handle_.get(), stream_));
            check_cublas_error(
                cublasSetPointerMode(handle_.get(), pointer_mode_));
            // insert the cublas handle in the arg list and call the cublas function
            detail::dispatch_helper<R, Params...> helper{};
            return helper(
                cublas_function, handle_.get(), std::forward<Args>(args)...);
        }

        // -------------------------------------------------------------------------
        // forward a cuda function through to the cuda executor base class
        // (we permit the use of a cublas executor for cuda calls)
        template <typename R, typename... Params, typename... Args>
        inline typename std::enable_if<std::is_same<cudaError_t, R>::value,
            void>::type
        apply(R (*cuda_function)(Params...), Args&&... args)
        {
            return cuda_executor::apply(
                cuda_function, std::forward<Args>(args)...);
        }

        // -------------------------------------------------------------------------
        // launch a cuBlas function and return a future that will become ready
        // when the task completes, this allows integration of GPU kernels with
        // hpx::futures and the tasking DAG.
        template <typename R, typename... Params, typename... Args>
        hpx::future<typename std::enable_if<
            std::is_same<cublasStatus_t, R>::value, void>::type>
        async(R (*cublas_function)(Params...), Args&&... args)
        {
            hpx::future<void> result;
            std::exception_ptr p;
            try
            {
                // make sue we run on the correct device
                check_cuda_error(cudaSetDevice(device_));
                // make sure this operation takes place on our stream
                check_cublas_error(cublasSetStream(handle_.get(), stream_));
                // insert the cublas handle in the arg list and call the cublas function
                detail::dispatch_helper<R, Params...> helper;
                helper(cublas_function, handle_.get(),
                    std::forward<Args>(args)...);
                return get_future();
            }
            catch (const hpx::exception& e)
            {
                p = std::current_exception();
            }

            // The exception is set outside the catch block since
            // set_exception may yield. Ending the catch block on a
            // different worker thread than where it was started may lead
            // to segfaults.
            auto state = traits::detail::get_shared_state(result);
            state->set_exception(std::move(p));
            return result;
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

        // return a copy of the cublas handle
        cublasHandle_t get_handle()
        {
            return handle_.get();
        }

    protected:
        handle_ptr handle_;
        cublasPointerMode_t pointer_mode_;
    };

}}}    // namespace hpx::cuda::experimental

namespace hpx { namespace parallel { namespace execution {
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
