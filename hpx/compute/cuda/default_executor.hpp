//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_CUDA_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA) && defined(__CUDACC__)
#include <hpx/lcos/future.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/executors/execution.hpp>

#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/cuda/default_executor_parameters.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/vector.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/functions.hpp>

namespace hpx { namespace compute { namespace cuda
{
    namespace detail
    {
        // generic implementation which simply passes through the shape elements
        template <typename Shape, typename Enable = void>
        struct bulk_launch_helper
        {
            template <typename F, typename ... Ts>
            static void call(cuda::target const& target, F && f,
                Shape const& shape, Ts &&... ts)
            {
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                std::size_t count =
                    std::distance(boost::begin(shape), boost::end(shape));
#else
                std::size_t count = boost::size(shape);
#endif

                int threads_per_block =
                    (std::min)(1024, static_cast<int>(count));
                int num_blocks = static_cast<int>(
                    (count + threads_per_block - 1) / threads_per_block);

                typedef typename boost::range_const_iterator<Shape>::type
                    iterator_type;
                typedef typename std::iterator_traits<iterator_type>::value_type
                    value_type;
                typedef cuda::allocator<value_type> alloc_type;

                // transfer shape to the GPU
                compute::vector<value_type, alloc_type> shape_container(
                    boost::begin(shape), boost::end(shape), alloc_type(target));

                detail::launch(
                    target, num_blocks, threads_per_block,
                    [] HPX_DEVICE
                        (F f, value_type* p, std::size_t count, Ts&... ts)
                    {
                        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < count)
                        {
                            hpx::util::invoke_r<void>(f, *(p + idx), ts...);
                        }
                    },
                    std::forward<F>(f), shape_container.data(), count,
                    std::forward<Ts>(ts)...);
            }
        };

        // specialization used by partitioner implementations
        template <typename Iterator>
        struct bulk_launch_helper<
            hpx::util::tuple<Iterator, std::size_t, std::size_t>,
            typename std::enable_if<
                hpx::traits::is_iterator<Iterator>::value
            >::type
        >
        {
            template <typename F, typename Shape, typename ... Ts>
            static void call(cuda::target const& target, F && f,
                Shape const& shape, Ts &&... ts)
            {
                typedef typename boost::range_const_iterator<Shape>::type
                    iterator_type;
                typedef typename std::iterator_traits<iterator_type>::value_type
                    value_type;

                for (auto const& s: shape)
                {
                    auto begin = hpx::util::get<0>(s);
                    std::size_t chunk_size = hpx::util::get<1>(s);

                    // FIXME: make the 1024 to be configurable...
                    int threads_per_block =
                        (std::min)(1024, static_cast<int>(chunk_size));
                    int num_blocks = static_cast<int>(
                        (chunk_size + threads_per_block - 1) / threads_per_block);

                    detail::launch(
                        target, num_blocks, threads_per_block,
                        [begin, chunk_size]
                        HPX_DEVICE (F f, Ts&... ts)
                        {
                            std::size_t idx
                                = blockIdx.x * blockDim.x + threadIdx.x;
                            if(idx < chunk_size)
                            {
                                hpx::util::invoke_r<void>(f,
                                    value_type(begin + idx, 1, idx), ts...);
                            }
                        },
                        std::forward<F>(f), std::forward<Ts>(ts)...
                    );
                }
            }
        };
    }

    struct default_executor
    {
        // By default, this executor relies on a special executor parameters
        // implementation which knows about the specifics of creating the
        // bulk-shape ranges for the accelerator.
        typedef default_executor_parameters executor_parameters_type;

        default_executor(cuda::target const& target)
          : target_(target)
        {}

        /// \cond NOINTERNAL
        bool operator==(default_executor const& rhs) const HPX_NOEXCEPT
        {
            return target_ == rhs.target_;
        }

        bool operator!=(default_executor const& rhs) const HPX_NOEXCEPT
        {
            return !(*this == rhs);
        }

        cuda::target const& context() const HPX_NOEXCEPT
        {
            return target_;
        }
        /// \endcond

        std::size_t processing_units_count() const
        {
            return target_.native_handle().processing_units();
        }

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts) const
        {
            detail::launch(target_, 1, 1,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<void> async_execute(F && f, Ts &&... ts) const
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            return target_.get_future();
        }

        template <typename F, typename ... Ts>
        void sync_execute(F && f, Ts &&... ts) const
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        template <typename F, typename Shape, typename ... Ts>
        void bulk_launch(F && f, Shape const& shape, Ts &&... ts) const
        {
            detail::bulk_launch_helper<Shape>::call(target_,
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<void> >
        async_bulk_execute(F && f, Shape const& shape, Ts &&... ts) const
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);

            std::vector<hpx::future<void> > result;
            result.push_back(target_.get_future());
            return result;
        }

        template <typename F, typename Shape, typename ... Ts>
        void sync_bulk_execute(F && f, Shape const& shape, Ts &&... ts) const
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        cuda::target& target()
        {
            return target_;
        }

        cuda::target const& target() const
        {
            return target_;
        }

    private:
        cuda::target target_;
    };
}}}

namespace hpx { namespace traits
{
    template <>
    struct executor_execution_category<compute::cuda::default_executor>
    {
        typedef parallel::execution::parallel_execution_tag type;
    };

    template <>
    struct is_one_way_executor<compute::cuda::default_executor>
      : std::true_type
    {};

    template <>
    struct is_two_way_executor<compute::cuda::default_executor>
      : std::true_type
    {};

    template <>
    struct is_bulk_one_way_executor<compute::cuda::default_executor>
      : std::true_type
    {};

    template <>
    struct is_bulk_two_way_executor<compute::cuda::default_executor>
      : std::true_type
    {};
}}

#endif
#endif
