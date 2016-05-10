///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
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
    template <typename F, typename Args>
    __global__ void launch_helper(F f, Args args)
    {
        // FIXME: is it possible to move tha arguments?
        hpx::util::invoke_fused(f, args);
    }

    // Launch any given function F with the given parameters. This function
    // does not involve any device synchronization.
    template <typename F, typename DimType, typename ...Ts>
    void launch(target const& t, DimType gridDim, DimType blockDim, F && f,
        Ts&&... vs)
    {
        detail::scoped_active_target active(t);

        typedef
            hpx::util::tuple<typename util::decay<Ts>::type...>
            args_type;
        typedef typename util::decay<F>::type fun_type;

        fun_type f_ = std::forward<F>(f);
        args_type args(std::forward<Ts>(vs)...);

        void (*launch_function)(fun_type, args_type)
            = launch_helper<fun_type, args_type>;

        launch_function<<<gridDim, blockDim, 0, active.stream()>>>(
            std::move(f_), std::move(args));

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::detail::launch()",
                std::string("kernel launch failed: ") +
                    cudaGetErrorString(error));
        }
    }
}}}}

#endif
#endif
