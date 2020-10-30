//  Copyright (c) 2017-2020 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/topology/topology.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_information_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v3 { namespace detail {
    /// \cond NOINTERNAL
    template <typename Parameters, typename Executor>
    std::size_t call_processing_units_parameter_count(
        Parameters&& params, Executor&& exec);
    /// \endcond
}}}}    // namespace hpx::parallel::v3::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution { namespace detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////
    // customization point for interface has_pending_closures()
    template <typename Executor>
    struct has_pending_closures_fn_helper<Executor,
        typename std::enable_if<
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value ||
            hpx::traits::is_never_blocking_one_way_executor<Executor>::value>::
            type>
    {
        template <typename AnyExecutor>
        HPX_FORCEINLINE static bool call(
            hpx::traits::detail::wrap_int, AnyExecutor&& /* exec */)
        {
            return false;    // assume stateless scheduling
        }

        template <typename AnyExecutor>
        HPX_FORCEINLINE static auto call(int, AnyExecutor&& exec)
            -> decltype(exec.has_pending_closures())
        {
            return exec.has_pending_closures();
        }

        template <typename AnyExecutor>
        struct result
        {
            using type = decltype(call(0, std::declval<AnyExecutor>()));
        };
    };

    ///////////////////////////////////////////////////////////////////////
    // customization point for interface get_pu_mask()
    template <typename Executor>
    struct get_pu_mask_fn_helper<Executor,
        typename std::enable_if<
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value ||
            hpx::traits::is_never_blocking_one_way_executor<Executor>::value>::
            type>
    {
        template <typename AnyExecutor>
        HPX_FORCEINLINE static threads::mask_cref_type call(
            hpx::traits::detail::wrap_int, AnyExecutor&& /* exec */,
            threads::topology& topo, std::size_t thread_num)
        {
            return get_pu_mask(topo, thread_num);
        }

        template <typename AnyExecutor>
        HPX_FORCEINLINE static auto call(int, AnyExecutor&& exec,
            threads::topology& topo, std::size_t thread_num)
            -> decltype(exec.get_pu_mask(topo, thread_num))
        {
            return exec.get_pu_mask(topo, thread_num);
        }

        template <typename AnyExecutor>
        struct result
        {
            using type = decltype(call(0, std::declval<AnyExecutor>(),
                std::declval<threads::topology&>(),
                std::declval<std::size_t>()));
        };
    };

    ///////////////////////////////////////////////////////////////////////
    // customization point for interface set_scheduler_mode()
    template <typename Executor>
    struct set_scheduler_mode_fn_helper<Executor,
        typename std::enable_if<
            hpx::traits::is_one_way_executor<Executor>::value ||
            hpx::traits::is_two_way_executor<Executor>::value ||
            hpx::traits::is_never_blocking_one_way_executor<Executor>::value>::
            type>
    {
        template <typename AnyExecutor, typename Mode>
        HPX_FORCEINLINE static void call(hpx::traits::detail::wrap_int,
            AnyExecutor&& /* exec */, Mode const& /* mode */)
        {
        }

        template <typename AnyExecutor, typename Mode>
        HPX_FORCEINLINE static auto call(int, AnyExecutor&& exec,
            Mode const& mode) -> decltype(exec.set_scheduler_mode(mode))
        {
            exec.set_scheduler_mode(mode);
        }

        template <typename AnyExecutor, typename Mode>
        struct result
        {
            using type = decltype(call(
                0, std::declval<AnyExecutor>(), std::declval<Mode const&>()));
        };
    };

    /// \endcond
}}}}    // namespace hpx::parallel::execution::detail
