//  Copyright (c) 2023-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_annotation.hpp

#pragma once

#include <hpx/modules/async_base.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <type_traits>

namespace hpx::execution::experimental {

    // Scheduling property implementations for execution policies that simply
    // forwards to the embedded executor

    HPX_CXX_CORE_EXPORT template <scheduling_property Tag,
        execution_policy ExPolicy, typename Property>
        requires(hpx::functional::is_tag_invocable_v<Tag,
            typename std::decay_t<ExPolicy>::executor_type, Property>)
    constexpr decltype(auto) tag_invoke(
        Tag tag, ExPolicy&& policy, Property&& prop)
    {
        return hpx::execution::experimental::create_rebound_policy(policy,
            tag(policy.executor(), HPX_FORWARD(Property, prop)),
            policy.parameters());
    }

    HPX_CXX_CORE_EXPORT template <scheduling_property Tag,
        execution_policy ExPolicy>
        requires(hpx::functional::is_tag_invocable_v<Tag,
            typename std::decay_t<ExPolicy>::executor_type>)
    constexpr decltype(auto) tag_invoke(Tag tag, ExPolicy&& policy)
    {
        return tag(policy.executor());
    }
}    // namespace hpx::execution::experimental
