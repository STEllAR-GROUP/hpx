//  Copyright (c) 2017-2021 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
#include <hpx/modules/topology.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    // Executor information customization points
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct has_pending_closures_fn_helper;

        template <typename Executor, typename Enable = void>
        struct get_pu_mask_fn_helper;

        template <typename Executor, typename Enable = void>
        struct set_scheduler_mode_fn_helper;
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // define customization points

    /// Retrieve whether this executor has operations pending or not.
    ///
    /// \param exec  [in] The executor object to use to extract the
    ///              requested information for.
    ///
    /// \note If the executor does not expose this information, this call
    ///       will always return \a false
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct has_pending_closures_t final
      : hpx::functional::tag_fallback<has_pending_closures_t>
    {
    private:
        // clang-format off
        template <typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            has_pending_closures_t, Executor&& exec)
        {
            return detail::has_pending_closures_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec));
        }
    } has_pending_closures{};

    /// Retrieve the bitmask describing the processing units the given
    /// thread is allowed to run on
    ///
    /// All threads::executors invoke sched.get_pu_mask().
    ///
    /// \param exec  [in] The executor object to use for querying the
    ///              number of pending tasks.
    /// \param topo  [in] The topology object to use to extract the
    ///              requested information.
    /// \param thream_num [in] The sequence number of the thread to
    ///              retrieve information for.
    ///
    /// \note If the executor does not support this operation, this call
    ///       will always invoke hpx::threads::get_pu_mask()
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct get_pu_mask_t final
      : hpx::functional::tag_fallback<get_pu_mask_t>
    {
    private:
        // clang-format off
        template <typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(get_pu_mask_t,
            Executor&& exec, threads::topology& topo, std::size_t thread_num)
        {
            return detail::get_pu_mask_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec), topo, thread_num);
        }
    } get_pu_mask{};

    /// Set various modes of operation on the scheduler underneath the
    /// given executor.
    ///
    /// \param exec     [in] The executor object to use.
    /// \param mode     [in] The new mode for the scheduler to pick up
    ///
    /// \note This calls exec.set_scheduler_mode(mode) if it exists;
    ///       otherwise it does nothing.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct set_scheduler_mode_t final
      : hpx::functional::tag_fallback<set_scheduler_mode_t>
    {
    private:
        // clang-format off
        template <typename Executor, typename Mode,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_invoke(
            set_scheduler_mode_t, Executor&& exec, Mode const& mode)
        {
            return detail::set_scheduler_mode_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec), mode);
        }
    } set_scheduler_mode{};
}}}    // namespace hpx::parallel::execution
