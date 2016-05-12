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
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/detail/scoped_active_target.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>

#include <cuda_runtime.h>

#include <string>

namespace hpx { namespace compute { namespace cuda { namespace detail
{
    template <typename Closure>
    __global__ void launch_function(Closure closure)
    {
        closure();
    }

    template <typename F, typename ...Ts>
    struct launch_helper
    {
        typedef
            hpx::util::tuple<typename util::decay<Ts>::type...>
            args_type;
        typedef typename util::decay<F>::type fun_type;


        struct closure
        {
            fun_type f_;
            args_type args_;
            HPX_DEVICE void operator()()
            {
                // FIXME: is it possible to move tha arguments?
                hpx::util::invoke_fused(f_, args_);
            }
        };

        typedef void (*launch_function_type)(closure);

        HPX_HOST_DEVICE
        static launch_function_type get_launch_function()
        {
            return launch_function<closure>;
        }

        template <typename DimType>
        HPX_HOST_DEVICE
        static void call(target const& tgt, DimType gridDim, DimType blockDim,
            fun_type f, args_type args)
        {
            // This is needed for the device code to make sure the kernel
            // is instantiated correctly.
            launch_function_type launcher = get_launch_function();
            closure c{std::move(f), std::move(args)};

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
            void *param_buffer
                = cudaGetParameterBuffer(std::alignment_of<closure>::value, sizeof(closure));
            std::memcpy(param_buffer, &c, sizeof(closure));
            cudaLaunchDevice(reinterpret_cast<void*>(launcher), param_buffer,
                dim3(gridDim), dim3(blockDim), 0, tgt.native_handle().stream_);
#else

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
        typedef
            hpx::util::tuple<typename util::decay<Ts>::type...>
            args_type;
        launch_helper<F, Ts...>::call(t, gridDim, blockDim, std::forward<F>(f),
            util::forward_as_tuple(std::forward<Ts>(vs)...));
    }
}}}}

#endif
#endif
