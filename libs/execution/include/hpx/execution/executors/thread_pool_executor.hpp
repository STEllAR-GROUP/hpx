//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_THREAD_POOL_EXECUTOR_HPP)
#define HPX_THREAD_POOL_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assertion.hpp>
#include <hpx/async_launch_policy_dispatch.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/post_policy_dispatch.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/util/unwrap.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a thread_pool_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    struct thread_pool_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        explicit thread_pool_executor(threads::thread_pool_base* pool)
          : pool_(pool)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(thread_pool_executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_;
        }

        bool operator!=(thread_pool_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        thread_pool_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<decltype(
                hpx::launch::async)>::call(hpx::launch::async, pool_,
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
            then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = hpx::util::one_shot(hpx::util::bind_back(
                std::forward<F>(f), std::forward<Ts>(ts)...));

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = lcos::detail::make_continuation_alloc_nounwrap<result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), hpx::launch::async,
                    std::move(func));

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                std::move(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            hpx::util::thread_description desc(
                f, "hpx::parallel::execution::parallel_executor::post");

            detail::post_policy_dispatch<decltype(hpx::launch::async)>::call(
                hpx::launch::async, desc, pool_,
                threads::thread_schedule_hint{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            std::size_t const os_thread_count = pool_->get_os_thread_count();
            hpx::util::thread_description const desc(f,
                "hpx::parallel::execution::thread_pool_executor::bulk_async_"
                "execute");

            typedef std::vector<hpx::future<
                typename detail::bulk_function_result<F, S, Ts...>::type>>
                result_type;

            result_type results;
            std::size_t const size = hpx::util::size(shape);
            results.resize(size);

            lcos::local::latch l(os_thread_count);
            std::size_t part_begin = 0;
            auto it = std::begin(shape);
            for (std::size_t t = 0; t < os_thread_count; ++t)
            {
                std::size_t const part_end = ((t + 1) * size) / os_thread_count;
                threads::thread_schedule_hint hint{
                    static_cast<std::int16_t>(t)};
                detail::post_policy_dispatch<decltype(
                    hpx::launch::async)>::call(hpx::launch::async, desc, pool_,
                    hint,

                    [&, this, hint, part_begin, part_end, f, it]() mutable {
                        for (std::size_t part_i = part_begin; part_i < part_end;
                             ++part_i)
                        {
                            results[part_i] =
                                hpx::detail::async_launch_policy_dispatch<
                                    decltype(hpx::launch::async)>::
                                    call(hpx::launch::async, pool_, hint, f,
                                        *it, ts...);
                            ++it;
                        }
                        l.count_down(1);
                    });
                std::advance(it, part_end - part_begin);
                part_begin = part_end;
            }

            l.wait();

            return results;
        }

        template <typename F, typename S, typename Future, typename... Ts>
        hpx::future<typename detail::bulk_then_execute_result<F, S, Future,
            Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            using func_result_type =
                typename detail::then_bulk_function_result<F, S, Future,
                    Ts...>::type;

            // std::vector<future<func_result_type>>
            using result_type = std::vector<hpx::future<func_result_type>>;

            auto&& func =
                detail::make_fused_bulk_async_execute_helper<result_type>(*this,
                    std::forward<F>(f), shape,
                    hpx::util::make_tuple(std::forward<Ts>(ts)...));

            // void or std::vector<func_result_type>
            using vector_result_type =
                typename detail::bulk_then_execute_result<F, S, Future,
                    Ts...>::type;

            // future<vector_result_type>
            using result_future_type = hpx::future<vector_result_type>;

            using shared_state_type =
                typename hpx::traits::detail::shared_state_ptr<
                    vector_result_type>::type;

            using future_type = typename std::decay<Future>::type;

            // vector<future<func_result_type>> -> vector<func_result_type>
            shared_state_type p =
                lcos::detail::make_continuation_alloc<vector_result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), hpx::launch::async,
                    [func = std::move(func)](future_type&& predecessor) mutable
                    -> vector_result_type {
                        // use unwrap directly (instead of lazily) to avoid
                        // having to pull in dataflow
                        return hpx::util::unwrap(func(std::move(predecessor)));
                    });

            return hpx::traits::future_access<result_future_type>::create(
                std::move(p));
            return hpx::make_ready_future();
        }
        /// \endcond

    private:
        threads::thread_pool_base* pool_;
    };
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<parallel::execution::thread_pool_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parallel::execution::thread_pool_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<parallel::execution::thread_pool_executor>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
