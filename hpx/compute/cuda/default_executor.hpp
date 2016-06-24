//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/vector.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <boost/range/functions.hpp>

namespace hpx { namespace compute { namespace cuda
{
    struct default_executor : hpx::parallel::executor_tag
    {
        default_executor(cuda::target& target)
          : target_(target)
        {}

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            detail::launch(target_, 1, 1,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<void> async_execute(F && f, Ts &&... ts)
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            return target_.get_future();
        }

        template <typename F, typename ... Ts>
        void execute(F && f, Ts &&... ts)
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        std::size_t processing_units_count()
        {
            cudaDeviceProp props;
            cudaError_t error = cudaGetDeviceProperties(&props,
                target_.native_handle().get_device());
            if (error != cudaSuccess)
            {
                // report error
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::default_executor::processing_units_count()",
                    std::string("cudaGetDeviceProperties failed: ") +
                        cudaGetErrorString(error));
            }

            std::size_t mp = props.multiProcessorCount;
            switch(props.major)
            {
                case 2:
                    if(props.minor == 1) return mp * 48;
                    return mp * 32;
                case 3:
                    return mp * 192;
                case 5:
                    return mp * 128;
                default:
                    break;
            }
            return mp;
        }

        template <typename F, typename Shape, typename ... Ts>
        void bulk_launch(F && f, Shape const& shape, Ts &&... ts)
        {
            std::size_t count = boost::size(shape);

            int threads_per_block = (std::min)(1024, int(count));
            int num_blocks =
                int((count + threads_per_block - 1) / threads_per_block);

            typedef typename boost::range_const_iterator<Shape>::type
                iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef cuda::allocator<value_type> alloc_type;
            typedef compute::vector<value_type, alloc_type> shape_container_type;

            // transfer shape to the GPU
            shape_container_type shape_container(
                boost::begin(shape), boost::end(shape), alloc_type(target_));

            value_type const* p = &(*boost::begin(shape));
            detail::launch(
                target_, num_blocks, threads_per_block,
                [] HPX_DEVICE (F f, value_type * p,
                    std::size_t count, Ts&... ts)
                {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < count)
                    {
                        hpx::util::invoke(f, *(p + idx), std::forward<Ts>(ts)...);
                    }
                },
                std::forward<F>(f), shape_container.data(), count,
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<void> >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);

            std::vector<hpx::future<void> > result;
            result.push_back(target_.get_future());
            return result;
        }

        template <typename F, typename Shape, typename ... Ts>
        void bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            target_.synchronize();
        }

    private:
        cuda::target& target_;
    };
}}}

#endif
#endif
