//  Copyright (c) 2021-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_annotation.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/annotating_executor.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/properties.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // with_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    template <execution_policy ExPolicy>
        requires(std::invocable<hpx::execution::experimental::with_annotation_t,
            typename std::decay_t<ExPolicy>::executor_type, char const*>)
    constexpr decltype(auto) tag_invoke(
        hpx::execution::experimental::with_annotation_t, ExPolicy&& policy,
        char const* annotation)
    {
        auto exec = hpx::execution::experimental::with_annotation(
            policy.executor(), annotation);

        return hpx::execution::experimental::create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    template <execution_policy ExPolicy>
        requires(std::invocable<hpx::execution::experimental::with_annotation_t,
            typename std::decay_t<ExPolicy>::executor_type, std::string>)
    decltype(auto) tag_invoke(hpx::execution::experimental::with_annotation_t,
        ExPolicy&& policy, std::string annotation)
    {
        auto exec = hpx::execution::experimental::with_annotation(
            policy.executor(), HPX_MOVE(annotation));

        return hpx::execution::experimental::create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    // get_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    template <execution_policy ExPolicy>
        requires(std::invocable<hpx::execution::experimental::get_annotation_t,
            typename std::decay_t<ExPolicy>::executor_type>)
    constexpr decltype(auto) tag_invoke(
        hpx::execution::experimental::get_annotation_t, ExPolicy&& policy)
    {
        return hpx::execution::experimental::get_annotation(policy.executor());
    }
}    // namespace hpx::execution::experimental
