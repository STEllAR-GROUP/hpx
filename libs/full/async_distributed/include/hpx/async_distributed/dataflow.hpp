//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/naming.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/pack_traversal/pack_traversal_async.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {

    template <typename Policy, typename Action, typename Args>
    struct dataflow_return_impl</*IsAction=*/true, Policy, Action, Args>
    {
        using type = hpx::future<typename Action::result_type>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy>
    struct dataflow_dispatch_impl<true, Policy,
        std::enable_if_t<traits::is_launch_policy_v<Policy>>>
    {
        template <typename Allocator, typename Policy_, typename Component,
            typename Signature, typename Derived, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(Allocator const& alloc,
            Policy_&& policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&,
            hpx::id_type const& id, Ts&&... ts)
        {
            return detail::create_dataflow(alloc, HPX_FORWARD(Policy_, policy),
                Derived{}, id,
                traits::acquire_future_disp()(HPX_FORWARD(Ts, ts))...);
        }
    };

    template <typename FD>
    struct dataflow_dispatch_impl<true, FD,
        std::enable_if_t<!traits::is_launch_policy_v<FD> &&
            !traits::is_one_way_executor_v<FD> &&
            !traits::is_two_way_executor_v<FD>>>
    {
        template <typename Allocator, typename Component, typename Signature,
            typename Derived, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(Allocator const& alloc,
            hpx::actions::basic_action<Component, Signature, Derived> const&
                act,
            hpx::id_type const& id, Ts&&... ts)
        {
            return dataflow_dispatch_impl<true, launch>::call(
                alloc, launch::async, act, id, HPX_FORWARD(Ts, ts)...);
        }
    };
}}}    // namespace hpx::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    // clang-format off
    template <typename Action, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            traits::is_action_v<Action> &&
           !traits::is_launch_policy_v<F>
        )>
    HPX_DEPRECATED_V(1, 9,
        "hpx::dataflow<Action>(...) is deprecated, use hpx::dataflow(Action{}, "
        "...) instead")
    // clang-format on
    decltype(auto) dataflow(F&& f, Ts&&... ts)
    {
        return hpx::detail::dataflow(hpx::util::internal_allocator<>{},
            hpx::launch::async, Action{}, HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }

    // clang-format off
    template <typename Action, typename F, typename... Ts,
        HPX_CONCEPT_REQUIRES_(
            traits::is_action_v<Action> &&
            traits::is_launch_policy_v<F>
        )>
    HPX_DEPRECATED_V(1, 9,
        "hpx::dataflow<Action>(policy, ...) is deprecated, use "
        "hpx::dataflow(policy, Action{}, ...) instead")
    // clang-format on
    decltype(auto) dataflow(F&& f, Ts&&... ts)
    {
        return hpx::detail::dataflow(hpx::util::internal_allocator<>{},
            HPX_FORWARD(F, f), Action{}, HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx
