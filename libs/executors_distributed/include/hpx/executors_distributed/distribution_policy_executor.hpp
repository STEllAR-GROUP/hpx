//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/distribution_policy_executor.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/runtime/components/server/invoke_function.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename F, bool IsAction, typename... Ts>
        struct distribution_policy_execute_result_impl;

        template <typename F, typename... Ts>
        struct distribution_policy_execute_result_impl<F, false, Ts...>
        {
            typedef typename hpx::util::detail::invoke_deferred_result<F,
                Ts...>::type type;
        };

        template <typename Action, typename... Ts>
        struct distribution_policy_execute_result_impl<Action, true, Ts...>
        {
            typedef
                typename hpx::util::decay<Action>::type::local_result_type type;
        };

        template <typename F, typename... Ts>
        struct distribution_policy_execute_result
          : distribution_policy_execute_result_impl<F,
                hpx::traits::is_action<
                    typename hpx::util::decay<F>::type>::value,
                Ts...>
        {
        };
    }    // namespace detail

    /// A \a distribution_policy_executor creates groups of parallel execution
    /// agents which execute in threads implicitly created by the executor and
    /// placed on any of the associated localities.
    ///
    /// \tparam  DistPolicy     The distribution policy type for which an
    ///          executor should be created. The expression
    ///          \a hpx::traits::is_distribution_policy<DistPolicy>::value must
    ///         evaluate to true.
    ///
    template <typename DistPolicy>
    class distribution_policy_executor
    {
    private:
        /// \cond NOINTERNAL
        static_assert(hpx::traits::is_distribution_policy<DistPolicy>::value,
            "distribution_policy_executor needs to be instantiated with a "
            "distribution policy type");

        // post implementations
        template <typename F, typename... Ts>
        typename std::enable_if<!hpx::traits::is_action<F>::value>::type
        post_impl(F&& f, Ts&&... ts) const
        {
            typedef components::server::invoke_function_action<F> action_type;
            policy_.template apply<action_type>(
                threads::thread_priority_default, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename Action, typename... Ts>
        typename std::enable_if<hpx::traits::is_action<Action>::value>::type
        post_impl(Action&& act, Ts&&... ts) const
        {
            policy_.template apply<Action>(
                threads::thread_priority_default, std::forward<Ts>(ts)...);
        }

        // async_execute implementations
        template <typename F, typename... Ts>
        typename std::enable_if<!hpx::traits::is_action<F>::value,
            hpx::future<typename detail::distribution_policy_execute_result<
                F>::type>>::type
        async_execute_impl(F&& f, Ts&&... ts) const
        {
            typedef components::server::invoke_function_action<F> action_type;
            return policy_.template async<action_type>(
                launch::async, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Action, typename... Ts>
        typename std::enable_if<hpx::traits::is_action<Action>::value,
            hpx::future<typename Action::local_result_type>>::type
        async_execute_impl(Action&& act, Ts&&... ts) const
        {
            return policy_.template async<Action>(
                launch::async, std::forward<Ts>(ts)...);
        }
        /// \endcond

    public:
        /// Create a new distribution_policy executor from the given
        /// distribution policy
        ///
        /// \param policy   The distribution_policy to create an executor from
        ///
        template <typename DistPolicy_,
            typename Enable = typename std::enable_if<
                !std::is_same<distribution_policy_executor,
                    typename hpx::util::decay<DistPolicy_>::type>::value>::type>
        distribution_policy_executor(DistPolicy_&& policy)
          : policy_(std::forward<DistPolicy_>(policy))
        {
        }

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
        typedef parallel_execution_tag execution_category;

        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts) const
        {
            return post_impl(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        hpx::future<
            typename detail::distribution_policy_execute_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts) const
        {
            return async_execute_impl(
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename... Ts>
        typename detail::distribution_policy_execute_result<F, Ts...>::type
        sync_execute(F&& f, Ts&&... ts) const
        {
            return async_execute_impl(
                std::forward<F>(f), std::forward<Ts>(ts)...)
                .get();
        }
        /// \endcond

    private:
        DistPolicy policy_;
    };

    /// Create a new distribution_policy_executor from the given
    /// distribution_policy.
    ///
    /// \param policy   The distribution_policy to create an executor from
    ///
    template <typename DistPolicy>
    distribution_policy_executor<typename hpx::util::decay<DistPolicy>::type>
    make_distribution_policy_executor(DistPolicy&& policy)
    {
        typedef typename hpx::util::decay<DistPolicy>::type dist_policy_type;
        return distribution_policy_executor<dist_policy_type>(
            std::forward<DistPolicy>(policy));
    }

    /// \cond NOINTERNAL
    template <typename DistPolicy>
    struct is_two_way_executor<
        parallel::execution::distribution_policy_executor<DistPolicy>>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
