//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2020 Teodor Nikolov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>
#include <hpx/async_cuda/cuda_future.hpp>
#include <hpx/async_cuda/target.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>

// CUDA runtime
#include <cuda_runtime.h>
//
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace cuda { namespace experimental {

    namespace detail {
        // -------------------------------------------------------------------------
        // A helper object to call a cudafunction returning a cudaError type
        // or a plain kernel definition (or cublas function in cublas executor)
        template <typename R, typename... Args>
        struct dispatch_helper;

        // default implementation - call the function
        template <typename R, typename... Args>
        struct dispatch_helper
        {
            inline R operator()(R (*f)(Args...), Args... args)
            {
                return f(args...);
            }
        };

        // specialization for return type void
        template <typename... Args>
        struct dispatch_helper<void, Args...>
        {
            inline void operator()(void (*f)(Args...), Args... args)
            {
                f(args...);
            }
        };

        // specialization for return type of cudaError_t
        template <typename... Args>
        struct dispatch_helper<cudaError_t, Args...>
        {
            inline cudaError_t operator()(
                cudaError_t (*f)(Args...), Args... args)
            {
                cudaError_t err = f(args...);
                return check_cuda_error(err);
            }
        };

    }    // namespace detail

    // -------------------------------------------------------------------------
    // Allows the launching of cuda functions and kernels on a stream with futures
    // returned that are set when the async functions/kernels are ready
    // -------------------------------------------------------------------------
    struct cuda_executor_base
    {
        using future_type = hpx::future<void>;

        // -------------------------------------------------------------------------
        // constructors - create a cuda stream that all tasks invoked by
        // this helper will use
        // assume event mode is the default
        cuda_executor_base(std::size_t device, bool event_mode)
          : device_(device)
          , event_mode_(event_mode)
        {
            target_ = std::make_shared<hpx::cuda::experimental::target>(device);
            stream_ = target_->native_handle().get_stream();
        }

        inline future_type get_future()
        {
            if (event_mode_)
            {
                return target_->get_future_with_event();
            }
            return target_->get_future_with_callback();
        }

    protected:
        int device_;
        bool event_mode_;
        cudaStream_t stream_;
        std::shared_ptr<hpx::cuda::experimental::target> target_;
    };

    // -------------------------------------------------------------------------
    // Allows you to launch kernels on a stream and get
    // futures back when they are ready
    // -------------------------------------------------------------------------
    struct cuda_executor : cuda_executor_base
    {
        // -------------------------------------------------------------------------
        // construct - create a cuda stream that all tasks invoked by
        // this helper will use
        cuda_executor(std::size_t device, bool event_mode = true)
          : cuda_executor_base(device, event_mode)
        {
        }

        // -------------------------------------------------------------------------
        // target destructor will clean up stream handle
        ~cuda_executor() {}

        // -------------------------------------------------------------------------
        // OneWay Execution
        template <typename F, typename... Ts>
        inline decltype(auto) post(F&& f, Ts&&... ts)
        {
            return apply(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // -------------------------------------------------------------------------
        // TwoWay Execution
        template <typename F, typename... Ts>
        inline decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return async(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

    protected:
        // -------------------------------------------------------------------------
        // launch a kernel on our stream and return without a future
        // the return value is the value returned from the cuda call
        // (typically this will be cudaError_t).
        // Throws cuda_exception if the async launch fails.
        template <typename R, typename... Params, typename... Args>
        void apply(R (*cuda_function)(Params...), Args&&... args)
        {
            // make sure we run on the correct device
            check_cuda_error(cudaSetDevice(device_));
            // insert the stream handle in the arg list and call the cuda function
            detail::dispatch_helper<R, Params...> helper{};
            helper(cuda_function, std::forward<Args>(args)..., stream_);
        }

        // -------------------------------------------------------------------------
        // launch a kernel on our stream and return a future that will become ready
        // when the task completes, this allows integregration of GPU kernels with
        // hpx::futures and the tasking DAG.
        // Puts a cuda_exception in the future if the async launch fails.
        template <typename R, typename... Params, typename... Args>
        hpx::future<void> async(R (*cuda_kernel)(Params...), Args&&... args)
        {
            hpx::future<void> result;
            std::exception_ptr p;
            try
            {
                // make sure we run on the correct device
                check_cuda_error(cudaSetDevice(device_));
                // insert the stream handle in the arg list and call the cuda function
                detail::dispatch_helper<R, Params...> helper{};
                helper(cuda_kernel, std::forward<Args>(args)..., stream_);
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
    /// \endcond
}}}    // namespace hpx::parallel::execution
