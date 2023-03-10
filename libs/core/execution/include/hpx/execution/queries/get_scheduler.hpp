//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/queries/read.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>

namespace hpx::execution::experimental {

    // [exec.sched_queries.forwarding_scheduler_query]
    //
    // 1. `execution::forwarding_scheduler_query` is used to ask a customization
    //    point object whether it is a scheduler query that should be forwarded
    //    through scheduler adaptors.
    //
    //
    // 2. The name `execution::forwarding_scheduler_query` denotes a
    //    customization point object. For some subexpression `t`,
    //    `execution::forwarding_scheduler_query(t)` is expression equivalent
    //    to:
    //
    //      1. `tag_invoke(execution::forwarding_scheduler_query, t)`,
    //         contextually converted to bool, if the tag_invoke expression is
    //         well formed.
    //
    //          - Mandates: The tag_invoke expression is indeed
    //                      contextually convertible to bool, that expression
    //                      and the contextual conversion are not potentially-
    //                      throwing and are core constant expressions if t is
    //                      a core constant expression.
    //
    //     2. Otherwise, false.
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct
        forwarding_scheduler_query_t final
      : hpx::functional::detail::tag_fallback_noexcept<
            forwarding_scheduler_query_t,
            detail::contextually_convertible_to_bool<
                forwarding_scheduler_query_t>>
    {
    private:
        template <typename T>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            forwarding_scheduler_query_t, T&&) noexcept
        {
            return false;
        }
    } forwarding_scheduler_query{};

    enum class forward_progress_guarantee
    {
        concurrent,
        parallel,
        weakly_parallel
    };

    // 1. `execution::get_forward_progress_guarantee` is used to ask a scheduler
    //    about the forward progress guarantees of execution agents created by
    //    that scheduler.
    //
    // 2. The name `execution::get_forward_progress_guarantee` denotes a
    //    customization point object. For some subexpression s, let S be
    //    decltype((s)).
    //
    //      1. If S does not satisfy execution::scheduler,
    //         execution::get_forward_progress_guarantee is ill-formed. Otherwise,
    //         execution::get_forward_progress_guarantee(s) is expression equivalent to:
    //      2. `tag_invoke(execution::get_forward_progress_guarantee, as_const(s))`,
    //         if this expression is well formed.
    //
    //          - Mandates: The tag_invoke expression above is not
    //                      potentially throwing and its type is
    //                      execution::forward_progress_guarantee.
    //
    //          Otherwise, execution::forward_progress_guarantee::weakly_parallel.
    //
    // 3. If `execution::get_forward_progress_guarantee(s)` for some scheduler s
    //    returns `execution::forward_progress_guarantee::concurrent`, all
    //    execution agents created by that scheduler shall provide the
    //    concurrent forward progress guarantee. If it returns
    //    `execution::forward_progress_guarantee::parallel`, all execution
    //    agents created by that scheduler shall provide at least the parallel
    //    forward progress guarantee.
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct
        get_forward_progress_guarantee_t final
      : hpx::functional::detail::tag_fallback_noexcept<
            get_forward_progress_guarantee_t>
    {
        template <typename T>
        friend constexpr HPX_FORCEINLINE forward_progress_guarantee
        tag_fallback_invoke(get_forward_progress_guarantee_t,
            hpx::util::unwrap_reference<T> const&) noexcept
        {
            return forward_progress_guarantee::weakly_parallel;
        }
    } get_forward_progress_guarantee{};

    // 1. execution::get_scheduler is used to ask an object for its associated
    //    scheduler.
    //
    // 2. The name execution::get_scheduler denotes a customization point object.
    //    For some subexpression r, if the type of r is (possibly cv-qualified)
    //    no_env, then execution::get_scheduler(r) is ill-formed.
    //
    //    Otherwise, it is expression equivalent to:
    //
    //    1. tag_invoke(execution::get_scheduler, as_const(r)), if this expression
    //       is well formed.
    //
    //       Mandates: The tag_invoke expression above is not potentially-
    //                 throwing and its type satisfies execution::scheduler.
    //
    //    2. Otherwise, execution::get_scheduler(r) is ill-formed.
    //
    // 3. execution::get_scheduler() (with no arguments) is expression-equivalent
    //    to execution::read(execution::get_scheduler).
    //
    inline constexpr struct get_scheduler_t final
      : hpx::functional::detail::tag_fallback<get_scheduler_t>
    {
    private:
        friend inline constexpr auto tag_fallback_invoke(
            get_scheduler_t) noexcept;

    } get_scheduler{};

    constexpr auto tag_fallback_invoke(get_scheduler_t) noexcept
    {
        return hpx::execution::experimental::read(get_scheduler);
    }
}    // namespace hpx::execution::experimental
