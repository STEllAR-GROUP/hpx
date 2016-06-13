///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_CUDA_DETAIL_LAUNCH_HPP
#define HPX_COMPUTE_CUDA_DETAIL_LAUNCH_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/compute/cuda/detail/scoped_active_target.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#include <cstring>
#endif
#include <string>

namespace hpx { namespace compute { namespace cuda { namespace detail
{
    template <typename Closure>
    __global__ void launch_function(Closure closure)
    {
        closure();
    }

    template <typename F, typename ...Ts>
    struct closure
    {
        typedef
            hpx::util::tuple<typename util::decay<Ts>::type...>
            args_type;
        typedef typename util::decay<F>::type fun_type;

        fun_type f_;
        args_type args_;

        HPX_HOST_DEVICE void operator()()
        {
            // FIXME: is it possible to move the arguments?
            hpx::util::invoke_fused(f_, args_);
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
        HPX_HOST_DEVICE
        static void call(target const& tgt, DimType gridDim, DimType blockDim,
            fun_type f, args_type args)
        {
            // This is needed for the device code to make sure the kernel
            // is instantiated correctly.
            launch_function_type launcher = get_launch_function();
            Closure c{std::move(f), std::move(args)};

            static_assert(sizeof(Closure) < 256,
                "We currently require the closure to be less than 256 bytes");

#if !defined(__CUDA_ARCH__)
            detail::scoped_active_target active(tgt);

            launch_function<<<gridDim, blockDim, 0, active.stream()>>>(
                std::move(c));

            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::detail::launch()",
                    std::string("kernel launch failed: ") +
                        cudaGetErrorString(error));
            }
#elif __CUDA_ARCH__ >= 350
            void *param_buffer = cudaGetParameterBuffer(
                std::alignment_of<Closure>::value, sizeof(Closure));
            std::memcpy(param_buffer, &c, sizeof(Closure));
            cudaLaunchDevice(reinterpret_cast<void*>(launcher), param_buffer,
                dim3(gridDim), dim3(blockDim), 0, tgt.native_handle().get_stream());
#endif
        }
    };


    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename DimType, typename F, typename ...Ts>
    HPX_HOST_DEVICE
    void launch(target const& t, DimType gridDim, DimType blockDim, F && f,
        Ts&&... vs)
    {
        typedef closure<F, Ts...> closure_type;
        launch_helper<closure_type>::call(t, gridDim, blockDim, std::forward<F>(f),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }
}}}}

#endif
#endif
