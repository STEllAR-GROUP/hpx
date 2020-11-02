//  Copyright (c) 2014-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file colocating_distribution_policy.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/async_distributed/detail/async_implementations.hpp>
#include <hpx/async_distributed/detail/async_unwrap_result_implementations.hpp>
#include <hpx/async_distributed/detail/sync_implementations.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    /// This class is a distribution policy that can be using with actions that
    /// return futures. For those actions it is possible to apply certain
    /// optimizations if the action is invoked synchronously.
    struct unwrapping_result_policy
    {
    public:
        explicit unwrapping_result_policy(id_type const& id)
          : id_(id)
        {}

        template <typename Client, typename Stub>
        explicit unwrapping_result_policy(
            client_base<Client, Stub> const& client)
          : id_(client.get_id())
        {}

        template <typename Action>
        struct async_result
        {
            using type = typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type;
        };

        template <typename Action, typename ...Ts>
        HPX_FORCEINLINE typename async_result<Action>::type
        async(launch policy, Ts&&... vs) const
        {
            return hpx::detail::async_unwrap_result_impl<Action>(
                policy, get_next_target(), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        HPX_FORCEINLINE typename async_result<Action>::type
        async(launch::sync_policy, Ts&&... vs) const
        {
            return hpx::detail::sync_impl<Action>(
                launch::sync, get_next_target(), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        typename async_result<Action>::type
        async_cb(launch policy, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::async_cb_impl<Action>(policy,
                get_next_target(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename... Ts>
        bool apply(Continuation&& c, threads::thread_priority priority,
            Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(std::forward<Continuation>(c),
                get_next_target(), priority, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename... Ts>
        bool apply(threads::thread_priority priority, Ts&&... vs) const
        {
            return hpx::detail::apply_impl<Action>(
                get_next_target(), priority, std::forward<Ts>(vs)...);
        }

        /// \note This function is part of the invocation policy implemented by
        ///       this class
        ///
        template <typename Action, typename Continuation, typename Callback,
            typename... Ts>
        bool apply_cb(Continuation&& c, threads::thread_priority priority,
            Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(
                std::forward<Continuation>(c), get_next_target(), priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename... Ts>
        bool apply_cb(
            threads::thread_priority priority, Callback&& cb, Ts&&... vs) const
        {
            return hpx::detail::apply_cb_impl<Action>(
                get_next_target(), priority,
                std::forward<Callback>(cb), std::forward<Ts>(vs)...);
        }

        hpx::id_type const& get_next_target() const
        {
            return id_;
        }

    protected:
        /// \cond NOINTERNAL
        hpx::id_type const& id_;   // locality to encapsulate
        /// \endcond
    };
}}

/// \cond NOINTERNAL
namespace hpx
{
    using unwrap_result = hpx::components::unwrapping_result_policy;

    namespace traits
    {
        template <>
        struct is_distribution_policy<components::unwrapping_result_policy>
          : std::true_type
        {};
    }
}
/// \endcond

