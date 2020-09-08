//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/executors/detail/hierarchical_spawning.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

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
    class thread_pool_executor
    {
        static constexpr std::size_t hierarchical_threshold_default_ = 6;

    public:
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        typedef hpx::execution::parallel_execution_tag execution_category;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef hpx::execution::static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        explicit thread_pool_executor(threads::thread_priority priority =
                                          threads::thread_priority_default,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default,
            threads::thread_schedule_hint schedulehint = {},
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(this_thread::get_pool())
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
            HPX_ASSERT(pool_);
        }

        explicit thread_pool_executor(
            threads::policies::scheduler_base* scheduler,
            threads::thread_priority priority =
                threads::thread_priority_default,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default,
            threads::thread_schedule_hint schedulehint = {},
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
            HPX_ASSERT(scheduler);
            pool_ = scheduler->get_parent_pool();
            HPX_ASSERT(pool_);
        }

        explicit thread_pool_executor(threads::thread_pool_base* pool,
            threads::thread_priority priority =
                threads::thread_priority_default,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize_default,
            threads::thread_schedule_hint schedulehint = {},
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(pool)
          , priority_(priority)
          , stacksize_(stacksize)
          , schedulehint_(schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
            HPX_ASSERT(pool_);
        }

        /// \cond NOINTERNAL
        bool operator==(thread_pool_executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(thread_pool_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        thread_pool_executor const& context() const noexcept
        {
            return *this;
        }

        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::detail::async_launch_policy_dispatch<decltype(
                launch::async)>::call(launch::async, pool_, priority_,
                stacksize_, schedulehint_, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
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
                p = hpx::lcos::detail::make_continuation_exec<result_type>(
                    std::forward<Future>(predecessor), *this, std::move(func));

            return hpx::traits::future_access<
                hpx::lcos::future<result_type>>::create(std::move(p));
        }

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            char const* annotation =
                hpx::traits::get_function_annotation<F>::call(f);
            hpx::util::thread_description desc(f, annotation);

            detail::post_policy_dispatch<decltype(launch::async)>::call(
                launch::async, desc, pool_, priority_, stacksize_,
                schedulehint_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<
            typename detail::bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            return detail::hierarchical_bulk_async_execute_helper(pool_,
                priority_, stacksize_, schedulehint_, 0,
                pool_->get_os_thread_count(), hierarchical_threshold_,
                launch::async, std::forward<F>(f), shape,
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        hpx::future<typename detail::bulk_then_execute_result<F, S, Future,
            Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return detail::hierarchical_bulk_then_execute_helper(*this,
                launch::async, std::forward<F>(f), shape,
                std::forward<Future>(predecessor), std::forward<Ts>(ts)...);
        }
        /// \endcond

    private:
        threads::thread_pool_base* pool_ = nullptr;
        threads::thread_priority priority_ = threads::thread_priority_default;
        threads::thread_stacksize stacksize_ =
            threads::thread_stacksize_default;
        threads::thread_schedule_hint schedulehint_ = {};
        std::size_t hierarchical_threshold_ = hierarchical_threshold_default_;
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
