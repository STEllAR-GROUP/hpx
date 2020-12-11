//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/futures/detail/future_transforms.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/acquire_future.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/naming.hpp>
#include <hpx/pack_traversal/pack_traversal_async.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/type_support/always_void.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>

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
        using type = hpx::lcos::future<typename Action::result_type>;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy>
    struct dataflow_dispatch_impl<true, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Allocator, typename Policy_, typename Component,
            typename Signature, typename Derived, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(Allocator const& alloc,
            Policy_&& policy,
            hpx::actions::basic_action<Component, Signature, Derived> const&
            /* act */,
            naming::id_type const& id, Ts&&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Policy_>(policy), Derived{}, id,
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }
    };

    template <typename FD>
    struct dataflow_dispatch_impl<true, FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || traits::is_threads_executor<FD>::value
#endif
                )>::type>
    {
        template <typename Allocator, typename Component, typename Signature,
            typename Derived, typename... Ts>
        HPX_FORCEINLINE static decltype(auto) call(Allocator const& alloc,
            hpx::actions::basic_action<Component, Signature, Derived> const&
                act,
            naming::id_type const& id, Ts&&... ts)
        {
            return dataflow_dispatch_impl<true, launch>::call(
                alloc, launch::async, act, id, std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename T0, typename Enable = void>
    struct dataflow_action_dispatch
    {
        template <typename Allocator, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Allocator const& alloc, naming::id_type const& id, Ts&&... ts)
        {
            return dataflow_dispatch_impl<true, Action>::call(
                alloc, Action(), id, std::forward<Ts>(ts)...);
        }
    };

    template <typename Action, typename Policy>
    struct dataflow_action_dispatch<Action, Policy,
        typename std::enable_if<traits::is_launch_policy<
            typename std::decay<Policy>::type>::value>::type>
    {
        template <typename Allocator, typename... Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<typename hpx::traits::
                    extract_action<Action>::remote_result_type>::type>
        call(Allocator const& alloc, Policy&& policy, naming::id_type const& id,
            Ts&&... ts)
        {
            return dataflow_dispatch_impl<true,
                typename std::decay<Policy>::type>::call(alloc,
                std::forward<Policy>(policy), Action(), id,
                std::forward<Ts>(ts)...);
        }
    };
}}}    // namespace hpx::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    template <typename Action, typename T0, typename... Ts,
        typename Enable =
            typename std::enable_if<traits::is_action<Action>::value>::type>
    HPX_FORCEINLINE auto dataflow(T0&& t0, Ts&&... ts)
        -> decltype(lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            hpx::util::internal_allocator<>{}, std::forward<T0>(t0),
            std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            hpx::util::internal_allocator<>{}, std::forward<T0>(t0),
            std::forward<Ts>(ts)...);
    }

    template <typename Action, typename Allocator, typename T0, typename... Ts,
        typename Enable =
            typename std::enable_if<traits::is_action<Action>::value>::type>
    HPX_FORCEINLINE auto dataflow_alloc(
        Allocator const& alloc, T0&& t0, Ts&&... ts)
        -> decltype(lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            alloc, std::forward<T0>(t0), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            alloc, std::forward<T0>(t0), std::forward<Ts>(ts)...);
    }
}    // namespace hpx

// #endif
