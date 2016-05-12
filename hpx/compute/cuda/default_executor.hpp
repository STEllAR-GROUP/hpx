//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>

#include <algorithm>
#include <utility>

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

//         template <typename F, typename Shape, typename ... Ts>
//         void bulk_launch(F && f, Shape const& shape, Ts &&... ts)
//         {
//             std::size_t count = boost::size(shape);
//
//             int threads_per_block = (std::min)(1024, int(count));
//             int num_blocks =
//                 int((count + threads_per_block - 1) / threads_per_block);
//
//             typedef typename boost::range_const_iterator<Shape>::type
//                 iterator_type;
//             typedef typename std::iterator_traits<iterator_type>::value_type
//                 value_type;
//
//             value_type const* p = &(*boost::begin(shape));
//             detail::launch(
//                 target_, num_blocks, threads_per_block,
//                 [] __device__ (F const& f, value_type const* p,
//                     std::size_t count, Ts const&... ts)
//                 {
//                     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//                     if (idx < count)
//                     {
// //                         std::advance(it, idx);
//                         hpx::util::invoke(f, *(p + idx), ts...);
//                     }
//                 },
//                 std::forward<F>(f), p, count, std::forward<Ts>(ts)...);
//         }
//
//         template <typename F, typename Shape, typename ... Ts>
//         std::vector<hpx::future<void> >
//         bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
//         {
//             bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
//
//             std::vector<hpx::future<void> > result;
//             result.push_back(target_.get_future());
//             return result;
//         }
//
//         template <typename F, typename Shape, typename ... Ts>
//         void bulk_execute(F && f, Shape const& shape, Ts &&... ts)
//         {
//             bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
//             target_.synchronize();
//         }

    private:
        cuda::target& target_;
    };
}}}

#endif
