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

namespace hpx::execution::experimental {

    // [exec.sched_queries.forwarding_scheduler_query]
    // 1. `execution::forwarding_scheduler_query` is used to ask a
    // customization point object whether it is a scheduler query that
    // should be forwarded through scheduler adaptors.
    // 2. The name `execution::forwarding_scheduler_query` denotes a
    // customization point object. For some subexpression `t`,
    // `execution::forwarding_scheduler_query(t)` is expression
    // equivalent to:
    //      1. `tag_invoke(execution::forwarding_scheduler_query, t)`,
    // contextually converted to bool, if the tag_invoke expression is
    // well formed.
    //          Mandates: The tag_invoke expression is indeed
    //          contextually convertible to bool, that expression and
    //          the contextual conversion are not potentially-throwing
    //          and are core constant expressions if t is a core
    //          constant expression.
    //     2. Otherwise, false.
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct
        forwarding_scheduler_query_t final
      : hpx::functional::detail::tag_fallback<forwarding_scheduler_query_t>
    {
    private:
        template <typename T>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            forwarding_scheduler_query_t, T&&) noexcept
        {
            return true;
        }
    } forwarding_scheduler_query{};

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
