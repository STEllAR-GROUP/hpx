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

    // 1. execution::get_delegatee_scheduler is used to ask an object for a
    //    scheduler that may be used to delegate work to for the purpose of
    //    forward progress delegation.
    //
    // 2. The name execution::get_scheduler denotes a customization point
    //    object.
    //    For some subexpression r, if the type of r is (possibly cv-qualified)
    //    no_env, then execution::get_delegatee_scheduler(r) is ill-formed.
    //    Otherwise, it is expression equivalent to:
    //
    //    1. tag_invoke(execution::get_delegatee_scheduler, as_const(r)), if
    //       this expression is well formed.
    //          Mandates: The tag_invoke expression above is not potentially-
    //          throwing and its type satisfies execution::scheduler.
    //
    //    2. Otherwise, execution::get_delegatee_scheduler(r) is ill-formed.
    //
    // 3. execution::get_delegatee_scheduler() (with no arguments) is expression-
    //    equivalent to execution::read(execution::get_delegatee_scheduler).
    //
    inline constexpr struct get_delegatee_scheduler_t final
      : hpx::functional::detail::tag_fallback<get_delegatee_scheduler_t>
    {
    private:
        friend inline constexpr auto tag_fallback_invoke(
            get_delegatee_scheduler_t) noexcept;

    } get_delegatee_scheduler{};

    constexpr auto tag_fallback_invoke(get_delegatee_scheduler_t) noexcept
    {
        return hpx::execution::experimental::read(get_delegatee_scheduler);
    }
}    // namespace hpx::execution::experimental
