//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_annotation.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/executors/annotating_executor.hpp>
#include <hpx/properties/property.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {

    // with_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    constexpr decltype(auto) tag_dispatch(
        hpx::execution::experimental::with_annotation_t, ExPolicy&& policy,
        char const* annotation)
    {
        auto exec = hpx::execution::experimental::with_annotation(
            policy.executor(), annotation);

        return hpx::parallel::execution::create_rebound_policy(
            policy, std::move(exec), policy.parameters());
    }

    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    decltype(auto) tag_dispatch(hpx::execution::experimental::with_annotation_t,
        ExPolicy&& policy, std::string annotation)
    {
        auto exec = hpx::execution::experimental::with_annotation(
            policy.executor(), std::move(annotation));

        return hpx::parallel::execution::create_rebound_policy(
            policy, std::move(exec), policy.parameters());
    }

    // get_annotation property implementation for execution policies
    // that simply forwards to the embedded executor
    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::functional::is_tag_dispatchable_v<
                hpx::execution::experimental::get_annotation_t,
                typename std::decay_t<ExPolicy>::executor_type>
        )>
    // clang-format on
    constexpr decltype(auto) tag_dispatch(
        hpx::execution::experimental::get_annotation_t, ExPolicy&& policy)
    {
        return hpx::execution::experimental::get_annotation(policy.executor());
    }
}}}    // namespace hpx::execution::experimental
