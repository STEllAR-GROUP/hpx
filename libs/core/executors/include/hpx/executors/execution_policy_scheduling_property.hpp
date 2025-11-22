//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_annotation.hpp

#pragma once

#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <type_traits>

namespace hpx::execution::experimental {

    // Scheduling property implementations for execution policies that simply
    // forwards to the embedded executor

    // clang-format off
    template <typename Tag, typename ExPolicy, typename Property,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag> &&
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::functional::is_tag_invocable_v<
                Tag, typename std::decay_t<ExPolicy>::executor_type,
                Property>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        Tag tag, ExPolicy&& policy, Property prop)
    {
        return hpx::execution::experimental::create_rebound_policy(
            policy, tag(policy.executor(), prop), policy.parameters());
    }

    // clang-format off
    template <typename Tag, typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag> &&
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::functional::is_tag_invocable_v<
                Tag, typename std::decay_t<ExPolicy>::executor_type>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(Tag tag, ExPolicy&& policy)
    {
        return tag(policy.executor());
    }
}    // namespace hpx::execution::experimental
