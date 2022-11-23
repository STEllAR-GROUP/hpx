//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_annotation.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/properties/property.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // with_hint property implementation for execution policies that simply
    // forwards to the embedded executor

    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::functional::is_tag_invocable_v<
                hpx::execution::experimental::with_hint_t,
                typename std::decay_t<ExPolicy>::executor_type,
                hpx::threads::thread_schedule_hint>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        hpx::execution::experimental::with_hint_t, ExPolicy&& policy,
        hpx::threads::thread_schedule_hint hint)
    {
        auto exec =
            hpx::execution::experimental::with_hint(policy.executor(), hint);

        return hpx::parallel::execution::create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    // get_hint property implementation for execution policies that simply
    // forwards to the embedded executor

    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::functional::is_tag_invocable_v<
                hpx::execution::experimental::get_hint_t,
                typename std::decay_t<ExPolicy>::executor_type>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        hpx::execution::experimental::get_hint_t, ExPolicy&& policy)
    {
        return hpx::execution::experimental::get_hint(policy.executor());
    }
}    // namespace hpx::execution::experimental
