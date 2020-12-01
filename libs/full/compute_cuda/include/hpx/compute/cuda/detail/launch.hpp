///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_GPU_SUPPORT)
#include <hpx/async_cuda/target.hpp>
#include <hpx/compute/cuda/detail/scoped_active_target.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/type_support/unused.hpp>

#include <hpx/async_cuda/custom_gpu_api.hpp>

#if __CUDA_ARCH__ >= 350
#include <cstring>
#endif
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace cuda { namespace experimental { namespace detail {
    template <typename Closure>
    __global__ void launch_function(Closure closure)
    {
        closure();
    }

    template <typename F, typename... Ts>
    struct closure
    {
        typedef hpx::tuple<typename std::decay<Ts>::type...> args_type;
        typedef typename std::decay<F>::type fun_type;

        fun_type f_;
        args_type args_;

        HPX_HOST_DEVICE closure(fun_type&& f, args_type&& args)
          : f_(std::move(f))
          , args_(std::move(args))
        {
        }

        HPX_HOST_DEVICE closure(closure const& rhs)
          : f_(rhs.f_)
          , args_(rhs.args_)
        {
        }

        HPX_HOST_DEVICE closure(closure&& rhs)
          : f_(std::move(rhs.f_))
          , args_(std::move(rhs.args_))
        {
        }

        closure& operator=(closure const&) = delete;

        HPX_DEVICE HPX_FORCEINLINE void operator()()
        {
            // FIXME: is it possible to move the arguments?
            hpx::util::invoke_fused_r<void>(f_, args_);
        }
    };

    template <typename Closure>
    struct launch_helper
    {
        typedef typename Closure::fun_type fun_type;
        typedef typename Closure::args_type args_type;
        typedef void (*launch_function_type)(Closure);

        HPX_HOST_DEVICE
        static launch_function_type get_launch_function()
        {
            return launch_function<Closure>;
        }

        template <typename DimType>
        HPX_HOST_DEVICE static void call(
            hpx::cuda::experimental::target const& tgt, DimType grid_dim,
            DimType block_dim, fun_type f, args_type args)
        {
            // This is needed for the device code to make sure the kernel
            // is instantiated correctly.
            launch_function_type launcher = get_launch_function();
            HPX_UNUSED(launcher);
            Closure c{std::move(f), std::move(args)};

            static_assert(sizeof(Closure) < 256,
                "We currently require the closure to be less than 256 bytes");

#if defined(HPX_COMPUTE_HOST_CODE)
            detail::scoped_active_target active(tgt);

            launch_function<<<grid_dim, block_dim, 0, active.stream()>>>(
                std::move(c));

            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error, "cuda::detail::launch()",
                    std::string("kernel launch failed: ") +
                        cudaGetErrorString(error));
            }
#elif __CUDA_ARCH__ >= 350
            void* param_buffer = cudaGetParameterBuffer(
                std::alignment_of<Closure>::value, sizeof(Closure));
            std::memcpy(param_buffer, &c, sizeof(Closure));
            //             cudaLaunchKernel(reinterpret_cast<void*>(launcher),
            //                 dim3(grid_dim), dim3(block_dim), param_buffer, 0,
            //                 tgt.native_handle().get_stream());
            cudaLaunchDevice(reinterpret_cast<void*>(launcher), param_buffer,
                dim3(grid_dim), dim3(block_dim), 0,
                tgt.native_handle().get_stream());
#else
            HPX_UNUSED(tgt);
            HPX_UNUSED(grid_dim);
            HPX_UNUSED(block_dim);
#endif
        }
    };

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename DimType, typename F, typename... Ts>
    HPX_HOST_DEVICE void launch(hpx::cuda::experimental::target const& t,
        DimType grid_dim, DimType block_dim, F&& f, Ts&&... vs)
    {
        typedef closure<F, Ts...> closure_type;
        launch_helper<closure_type>::call(t, grid_dim, block_dim,
            std::forward<F>(f), hpx::forward_as_tuple(std::forward<Ts>(vs)...));
    }
}}}}    // namespace hpx::cuda::experimental::detail

#endif
