//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution/executors/static_chunk_size.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/detail/hierarchical_spawning.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { namespace execution { namespace detail {
    template <typename Policy>
    struct get_default_policy
    {
        static constexpr Policy call() noexcept
        {
            return Policy{};
        }
    };

    template <>
    struct get_default_policy<hpx::launch>
    {
        static constexpr hpx::launch::async_policy call() noexcept
        {
            return hpx::launch::async_policy{};
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename... Ts>
    struct bulk_function_result;

    ///////////////////////////////////////////////////////////////////////
    template <typename F, typename Shape, typename Future, typename... Ts>
    struct bulk_then_execute_result;

    template <typename F, typename Shape, typename Future, typename... Ts>
    struct then_bulk_function_result;
}}}}    // namespace hpx::parallel::execution::detail

namespace hpx { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    ///
    /// This executor conforms to the concepts of a TwoWayExecutor,
    /// and a BulkTwoWayExecutor
    template <typename Policy>
    struct parallel_policy_executor
    {
        /// Associate the parallel_execution_tag executor tag type as a default
        /// with this executor.
        using execution_category = parallel_execution_tag;

        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        using executor_parameters_type = static_chunk_size;

        /// Create a new parallel executor
        constexpr explicit parallel_policy_executor(
            threads::thread_priority priority,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call(),
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(nullptr)
          , policy_(l, priority, stacksize, schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(
                l, threads::thread_priority::default_, stacksize, schedulehint)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_schedule_hint schedulehint,
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(l, threads::thread_priority::default_,
                threads::thread_stacksize::default_, schedulehint)
        {
        }

        constexpr explicit parallel_policy_executor(
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call())
          : pool_(nullptr)
          , policy_(l)
        {
        }

        constexpr explicit parallel_policy_executor(
            threads::thread_pool_base* pool,
            threads::thread_priority priority =
                threads::thread_priority::default_,
            threads::thread_stacksize stacksize =
                threads::thread_stacksize::default_,
            threads::thread_schedule_hint schedulehint = {},
            Policy l =
                parallel::execution::detail::get_default_policy<Policy>::call(),
            std::size_t hierarchical_threshold =
                hierarchical_threshold_default_)
          : pool_(pool)
          , policy_(l, priority, stacksize, schedulehint)
          , hierarchical_threshold_(hierarchical_threshold)
        {
        }

        // property implementations
        friend constexpr parallel_policy_executor tag_dispatch(
            hpx::execution::experimental::with_hint_t,
            parallel_policy_executor const& exec,
            hpx::threads::thread_schedule_hint hint)
        {
            auto exec_with_hint = exec;
            exec_with_hint.policy_ = hint;
            return exec_with_hint;
        }

        friend constexpr hpx::threads::thread_schedule_hint tag_dispatch(
            hpx::execution::experimental::get_hint_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.policy_.hint();
        }

        friend constexpr parallel_policy_executor tag_dispatch(
            hpx::execution::experimental::with_priority_t,
            parallel_policy_executor const& exec,
            hpx::threads::thread_priority priority)
        {
            auto exec_with_priority = exec;
            exec_with_priority.policy_ = priority;
            return exec_with_priority;
        }

        friend constexpr hpx::threads::thread_priority tag_dispatch(
            hpx::execution::experimental::get_priority_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.policy_.priority();
        }

        friend constexpr parallel_policy_executor tag_dispatch(
            hpx::execution::experimental::with_annotation_t,
            parallel_policy_executor const& exec, char const* annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ = annotation;
            return exec_with_annotation;
        }

        friend parallel_policy_executor tag_dispatch(
            hpx::execution::experimental::with_annotation_t,
            parallel_policy_executor const& exec, std::string annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ =
                util::detail::store_function_annotation(std::move(annotation));
            return exec_with_annotation;
        }

        friend constexpr char const* tag_dispatch(
            hpx::execution::experimental::get_annotation_t,
            parallel_policy_executor const& exec) noexcept
        {
            return exec.annotation_;
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(
            parallel_policy_executor const& rhs) const noexcept
        {
            return policy_ == rhs.policy_ && pool_ == rhs.pool_ &&
                hierarchical_threshold_ == rhs.hierarchical_threshold_;
        }

        constexpr bool operator!=(
            parallel_policy_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr parallel_policy_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        sync_execute(F&& f, Ts&&... ts) const
        {
            hpx::util::annotate_function annotate(annotation_ ?
                    annotation_ :
                    "parallel_policy_executor::sync_execute");
            return hpx::detail::sync_launch_policy_dispatch<Policy>::call(
                launch::sync, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            hpx::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();

            return hpx::detail::async_launch_policy_dispatch<Policy>::call(
                policy_, desc, pool, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        HPX_FORCEINLINE
            hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
                Future, Ts...>::type>
            then_execute(F&& f, Future&& predecessor, Ts&&... ts) const
        {
            using result_type =
                typename hpx::util::detail::invoke_deferred_result<F, Future,
                    Ts...>::type;

            auto&& func = hpx::util::one_shot(hpx::util::bind_back(
                hpx::util::annotated_function(std::forward<F>(f), annotation_),
                std::forward<Ts>(ts)...));

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = lcos::detail::make_continuation_alloc_nounwrap<result_type>(
                    hpx::util::internal_allocator<>{},
                    std::forward<Future>(predecessor), policy_,
                    std::move(func));

            return hpx::traits::future_access<hpx::future<result_type>>::create(
                std::move(p));
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            hpx::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();
            parallel::execution::detail::post_policy_dispatch<Policy>::call(
                policy_, desc, pool, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        std::vector<hpx::future<typename parallel::execution::detail::
                bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts) const
        {
            hpx::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();
            return parallel::execution::detail::
                hierarchical_bulk_async_execute_helper(desc, pool, 0,
                    pool->get_os_thread_count(), hierarchical_threshold_,
                    policy_, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        hpx::future<typename parallel::execution::detail::
                bulk_then_execute_result<F, S, Future, Ts...>::type>
        bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            return parallel::execution::detail::
                hierarchical_bulk_then_execute_helper(*this, policy_,
                    hpx::util::annotated_function(
                        std::forward<F>(f), annotation_),
                    shape, std::forward<Future>(predecessor),
                    std::forward<Ts>(ts)...);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & policy_ & hierarchical_threshold_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        static constexpr std::size_t hierarchical_threshold_default_ = 6;

        threads::thread_pool_base* pool_;
        Policy policy_;
        std::size_t hierarchical_threshold_ = hierarchical_threshold_default_;
        char const* annotation_ = nullptr;
        /// \endcond
    };

    using parallel_executor = parallel_policy_executor<hpx::launch>;
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {
    using parallel_executor HPX_LOCAL_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_executor is deprecated. Use "
        "hpx::execution::parallel_executor instead.") =
        hpx::execution::parallel_executor;
    template <typename Policy>
    using parallel_policy_executor HPX_LOCAL_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_policy_executor is deprecated. Use "
        "hpx::execution::parallel_policy_executor instead.") =
        hpx::execution::parallel_policy_executor<Policy>;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <typename Policy>
    struct is_one_way_executor<hpx::execution::parallel_policy_executor<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_two_way_executor<hpx::execution::parallel_policy_executor<Policy>>
      : std::true_type
    {
    };

    template <typename Policy>
    struct is_bulk_two_way_executor<
        hpx::execution::parallel_policy_executor<Policy>> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
