//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_GPU_SUPPORT)
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/modules/errors.hpp>

#include <hpx/execution/executors/execution.hpp>

#include <hpx/async_cuda/target.hpp>
#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/compute/cuda/default_executor_parameters.hpp>
#include <hpx/compute/cuda/detail/launch.hpp>
#include <hpx/compute/vector.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace cuda { namespace experimental {
    namespace detail {
        // generic implementation which simply passes through the shape elements
        template <typename Shape, typename Enable = void>
        struct bulk_launch_helper
        {
            template <typename F, typename... Ts>
            static void call(hpx::cuda::experimental::target const& target,
                F&& f, Shape const& shape,
                Ts&&...
#if defined(HPX_COMPUTE_DEVICE_CODE) || defined(HPX_COMPUTE_HOST_CODE)
                ts
#endif
            )
            {
#if defined(HPX_COMPUTE_DEVICE_CODE) || defined(HPX_COMPUTE_HOST_CODE)
                std::size_t count = util::size(shape);

                int threads_per_block =
                    (std::min)(1024, static_cast<int>(count));
                int num_blocks = static_cast<int>(
                    (count + threads_per_block - 1) / threads_per_block);

                typedef typename hpx::traits::range_traits<Shape>::value_type
                    value_type;
                typedef cuda::experimental::allocator<value_type> alloc_type;

                // transfer shape to the GPU
                compute::vector<value_type, alloc_type> shape_container(
                    util::begin(shape), util::end(shape), alloc_type(target));

                detail::launch(
                    target, num_blocks, threads_per_block,
                    [] HPX_DEVICE(
                        F f, value_type * p, std::size_t count, Ts & ... ts) {
                        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < count)
                        {
                            hpx::util::invoke_r<void>(f, *(p + idx), ts...);
                        }
                    },
                    std::forward<F>(f), shape_container.data(), count,
                    std::forward<Ts>(ts)...);
#else
                HPX_UNUSED(target);
                HPX_UNUSED(f);
                HPX_UNUSED(shape);
                HPX_THROW_EXCEPTION(hpx::not_implemented,
                    "hpx::cuda::experimental::detail::bulk_launch_helper",
                    "Trying to launch a CUDA kernel, but did not compile in "
                    "CUDA mode");
#endif
            }
        };

        // specialization used by partitioner implementations
        template <typename Iterator>
        struct bulk_launch_helper<
            hpx::tuple<Iterator, std::size_t, std::size_t>,
            typename std::enable_if<
                hpx::traits::is_iterator<Iterator>::value>::type>
        {
            template <typename F, typename Shape, typename... Ts>
            static void call(hpx::cuda::experimental::target const& target,
                F&& f, Shape const& shape,
                Ts&&...
#if defined(HPX_COMPUTE_DEVICE_CODE) || defined(HPX_COMPUTE_HOST_CODE)
                ts
#endif
            )
            {
#if defined(HPX_COMPUTE_DEVICE_CODE) || defined(HPX_COMPUTE_HOST_CODE)
                typedef typename hpx::traits::range_traits<Shape>::value_type
                    value_type;

                for (auto const& s : shape)
                {
                    auto begin = hpx::get<0>(s);
                    std::size_t chunk_size = hpx::get<1>(s);

                    // FIXME: make the 1024 to be configurable...
                    int threads_per_block =
                        (std::min)(1024, static_cast<int>(chunk_size));
                    int num_blocks =
                        static_cast<int>((chunk_size + threads_per_block - 1) /
                            threads_per_block);

                    detail::launch(
                        target, num_blocks, threads_per_block,
                        [begin, chunk_size] HPX_DEVICE(F f, Ts & ... ts) {
                            std::size_t idx =
                                blockIdx.x * blockDim.x + threadIdx.x;
                            if (idx < chunk_size)
                            {
                                hpx::util::invoke_r<void>(
                                    f, value_type(begin + idx, 1, idx), ts...);
                            }
                        },
                        std::forward<F>(f), std::forward<Ts>(ts)...);
                }
#else
                HPX_UNUSED(target);
                HPX_UNUSED(f);
                HPX_UNUSED(shape);
                HPX_THROW_EXCEPTION(hpx::not_implemented,
                    "hpx::cuda::experimental::detail::bulk_launch_helper",
                    "Trying to launch a CUDA kernel, but did not compile in "
                    "CUDA mode");
#endif
            }
        };
    }    // namespace detail

    struct default_executor
    {
        // By default, this executor relies on a special executor parameters
        // implementation which knows about the specifics of creating the
        // bulk-shape ranges for the accelerator.
        typedef default_executor_parameters executor_parameters_type;

        default_executor(hpx::cuda::experimental::target const& target)
          : target_(target)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(default_executor const& rhs) const noexcept
        {
            return target_ == rhs.target_;
        }

        bool operator!=(default_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        hpx::cuda::experimental::target const& context() const noexcept
        {
            return target_;
        }
        /// \endcond

        std::size_t processing_units_count() const
        {
            return target_.native_handle().processing_units();
        }

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            detail::launch(
                target_, 1, 1, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        hpx::future<void> async_execute(F&& f, Ts&&... ts) const
        {
            post(std::forward<F>(f), std::forward<Ts>(ts)...);
            return target_.get_future_with_callback();
        }

        template <typename F, typename... Ts>
        void sync_execute(F&& f, Ts&&... ts) const
        {
            post(std::forward<F>(f), std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        template <typename F, typename Shape, typename... Ts>
        void bulk_launch(F&& f, Shape const& shape, Ts&&... ts) const
        {
            detail::bulk_launch_helper<Shape>::call(
                target_, std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename... Ts>
        std::vector<hpx::future<void>> bulk_async_execute(
            F&& f, Shape const& shape, Ts&&... ts) const
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);

            std::vector<hpx::future<void>> result;
            result.push_back(target_.get_future_with_callback());
            return result;
        }

        template <typename F, typename Shape, typename... Ts>
        void bulk_sync_execute(F&& f, Shape const& shape, Ts&&... ts) const
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        hpx::cuda::experimental::target& target()
        {
            return target_;
        }

        hpx::cuda::experimental::target const& target() const
        {
            return target_;
        }

    private:
        hpx::cuda::experimental::target target_;
    };
}}}    // namespace hpx::cuda::experimental

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct executor_execution_category<cuda::experimental::default_executor>
    {
        typedef ::hpx::execution::parallel_execution_tag type;
    };

    template <>
    struct is_one_way_executor<cuda::experimental::default_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<cuda::experimental::default_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_one_way_executor<cuda::experimental::default_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<cuda::experimental::default_executor>
      : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

#endif
