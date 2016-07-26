//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/distribution_policy_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_DISTRIBUTION_POLICY_EXECUTOR_JUL_21_2015_0404PM)
#define HPX_PARALLEL_EXECUTORS_DISTRIBUTION_POLICY_EXECUTOR_JUL_21_2015_0404PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/server/invoke_function.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_executor.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>

#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/result_of.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F, bool IsAction, typename ... Ts>
        struct distribution_policy_execute_result_impl;

        template <typename F, typename ... Ts>
        struct distribution_policy_execute_result_impl<F, false, Ts...>
        {
            typedef typename hpx::util::detail::deferred_result_of<
                    F(Ts...)
                >::type type;
        };

        template <typename Action, typename ... Ts>
        struct distribution_policy_execute_result_impl<Action, true, Ts...>
        {
            typedef typename hpx::util::decay<Action>::type::local_result_type
                type;
        };

        template <typename F, typename ... Ts>
        struct distribution_policy_execute_result
          : distribution_policy_execute_result_impl<F,
                hpx::traits::is_action<typename hpx::util::decay<F>::type>::value,
                Ts...>
        {};
    }

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
    class distribution_policy_executor : public executor_tag
    {
    private:
        /// \cond NOINTERNAL
        static_assert(
            hpx::traits::is_distribution_policy<DistPolicy>::value,
            "distribution_policy_executor needs to be instantiated with a "
                "distribution policy type");

        // apply_execute implementations
        template <typename F, typename ... Ts>
        typename std::enable_if<
            !hpx::traits::is_action<F>::value
        >::type
        apply_execute_impl(F && f, Ts && ... ts) const
        {
            typedef components::server::invoke_function_action<F> action_type;
            policy_.template apply<action_type>(threads::thread_priority_default,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Action, typename ... Ts>
        typename std::enable_if<
            hpx::traits::is_action<Action>::value
        >::type
        apply_execute_impl(Action && act, Ts && ... ts) const
        {
            policy_.template apply<Action>(threads::thread_priority_default,
                std::forward<Ts>(ts)...);
        }

        // async_execute implementations
        template <typename F, typename ... Ts>
        typename std::enable_if<
           !hpx::traits::is_action<F>::value,
            hpx::future<
                typename detail::distribution_policy_execute_result<F>::type
            >
        >::type
        async_execute_impl(F && f, Ts &&... ts) const
        {
            typedef components::server::invoke_function_action<F> action_type;
            return policy_.template async<action_type>(launch::async,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Action, typename ... Ts>
        typename std::enable_if<
            hpx::traits::is_action<Action>::value,
            hpx::future<typename Action::local_result_type>
        >::type
        async_execute_impl(Action && act, Ts &&... ts) const
        {
            return policy_.template async<Action>(launch::async,
                std::forward<Ts>(ts)...);
        }
        /// \endcond

    public:
        /// Create a new distribution_policy executor from the given
        /// distribution policy
        ///
        /// \param policy   The distribution_policy to create an executor from
        ///
        template <typename DistPolicy_>
        distribution_policy_executor(DistPolicy_ && policy)
          : policy_(std::forward<DistPolicy_>(policy))
        {}

        /// \cond NOINTERNAL
        typedef parallel_execution_tag execution_category;

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts) const
        {
            return apply_execute_impl(std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<
            typename detail::distribution_policy_execute_result<F&&, Ts&&...>::type
        >
        async_execute(F && f, Ts &&... ts) const
        {
            return async_execute_impl(std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        typename detail::distribution_policy_execute_result<F&&, Ts&&...>::type
        execute(F && f, Ts &&... ts) const
        {
            return async_execute_impl(std::forward<F>(f),
                std::forward<Ts>(ts)...).get();
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
    make_distribution_policy_executor(DistPolicy && policy)
    {
        typedef typename hpx::util::decay<DistPolicy>::type dist_policy_type;
        return distribution_policy_executor<dist_policy_type>(
            std::forward<DistPolicy>(policy));
    }

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename DistPolicy>
        struct is_executor<distribution_policy_executor<DistPolicy> >
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
