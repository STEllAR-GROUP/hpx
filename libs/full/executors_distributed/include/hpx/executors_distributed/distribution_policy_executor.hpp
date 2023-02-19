//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/distribution_policy_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/invoke_function.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/execution/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel::execution {

    namespace detail {

        template <typename F, bool IsAction, typename... Ts>
        struct distribution_policy_execute_result_impl;

        template <typename F, typename... Ts>
        struct distribution_policy_execute_result_impl<F, false, Ts...>
        {
            using type = hpx::util::detail::invoke_deferred_result_t<F, Ts...>;
        };

        template <typename Action, typename... Ts>
        struct distribution_policy_execute_result_impl<Action, true, Ts...>
        {
            using type = typename std::decay_t<Action>::local_result_type;
        };

        template <typename F, typename... Ts>
        struct distribution_policy_execute_result
          : distribution_policy_execute_result_impl<F,
                hpx::traits::is_action_v<std::decay_t<F>>, Ts...>
        {
        };

        template <typename F, typename... Ts>
        using distribution_policy_execute_result_t =
            typename distribution_policy_execute_result<F, Ts...>::type;
    }    // namespace detail

    /// A \a distribution_policy_executor creates groups of parallel execution
    /// agents that execute in threads implicitly created by the executor and
    /// placed on any of the associated localities.
    ///
    /// \tparam  DistPolicy     The distribution policy type for which an
    ///          executor should be created. The expression
    ///          \a hpx::traits::is_distribution_policy_v<DistPolicy> must
    ///         evaluate to true.
    ///
    template <typename DistPolicy>
    class distribution_policy_executor
    {
        /// \cond NOINTERNAL
        static_assert(hpx::traits::is_distribution_policy_v<DistPolicy>,
            "distribution_policy_executor needs to be instantiated with a "
            "distribution policy type");

        // post implementations
        template <typename F, typename... Ts>
        std::enable_if_t<!hpx::traits::is_action_v<F>> post_impl(
            F&& f, Ts&&... ts) const
        {
            using action_type =
                components::server::invoke_function_action<std::decay_t<F>,
                    std::decay_t<Ts>...>;

            policy_.template apply<action_type>(
                threads::thread_priority::default_, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename Action, typename... Ts>
        std::enable_if_t<hpx::traits::is_action_v<Action>> post_impl(
            Action&&, Ts&&... ts) const
        {
            policy_.template apply<Action>(
                threads::thread_priority::default_, HPX_FORWARD(Ts, ts)...);
        }

        // async_execute implementations
        template <typename F, typename... Ts>
        std::enable_if_t<!hpx::traits::is_action_v<F>,
            hpx::future<
                detail::distribution_policy_execute_result_t<F, Ts&&...>>>
        async_execute_impl(F&& f, Ts&&... ts) const
        {
            using action_type =
                components::server::invoke_function_action<std::decay_t<F>,
                    std::decay_t<Ts>...>;

            return policy_.template async<action_type>(
                launch::async, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename Action, typename... Ts>
        std::enable_if_t<hpx::traits::is_action_v<Action>,
            hpx::future<typename Action::local_result_type>>
        async_execute_impl(Action&&, Ts&&... ts) const
        {
            return policy_.template async<Action>(
                launch::async, HPX_FORWARD(Ts, ts)...);
        }
        /// \endcond

    public:
        /// Create a new distribution_policy executor from the given
        /// distribution policy
        ///
        /// \param policy   The distribution_policy to create an executor from
        ///
        template <typename DistPolicy_,
            typename Enable = std::enable_if_t<!std::is_same_v<
                distribution_policy_executor, std::decay_t<DistPolicy_>>>>
        explicit distribution_policy_executor(DistPolicy_&& policy)
          : policy_(HPX_FORWARD(DistPolicy_, policy))
        {
        }

        template <typename T>
        using future_type = hpx::future<T>;

        /// \cond NOINTERNAL
        bool operator==(distribution_policy_executor const& rhs) const noexcept
        {
            return policy_ == rhs.policy_;
        }

        bool operator!=(distribution_policy_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        distribution_policy_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL
        using execution_category = hpx::execution::parallel_execution_tag;

    private:
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
            distribution_policy_executor const& exec, F&& f, Ts&&... ts)
        {
            return exec.post_impl(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            distribution_policy_executor const& exec, F&& f, Ts&&... ts)
        {
            return exec.async_execute_impl(
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::sync_execute_t,
            distribution_policy_executor const& exec, F&& f, Ts&&... ts)
        {
            return exec
                .async_execute_impl(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)
                .get();
        }
        /// \endcond

    private:
        DistPolicy policy_;
    };

    template <typename DistPolicy>
    distribution_policy_executor(DistPolicy&&)
        -> distribution_policy_executor<std::decay_t<DistPolicy>>;

    /// Create a new distribution_policy_executor from the given
    /// distribution_policy.
    ///
    /// \param policy   The distribution_policy to create an executor from
    ///
    template <typename DistPolicy>
    HPX_DEPRECATED_V(1, 9,
        "hpx::parallel::execution::make_distribution_policy_executor is "
        "deprecated, use "
        "hpx::parallel::execution::distribution_policy_executor instead")
    distribution_policy_executor<std::decay_t<
        DistPolicy>> make_distribution_policy_executor(DistPolicy&& policy)
    {
        using dist_policy_type = std::decay_t<DistPolicy>;
        return distribution_policy_executor<dist_policy_type>(
            HPX_FORWARD(DistPolicy, policy));
    }

    /// \cond NOINTERNAL
    template <typename DistPolicy>
    struct is_two_way_executor<
        parallel::execution::distribution_policy_executor<DistPolicy>>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
