//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_parameters.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/tag_invoke.hpp>

#include <concepts>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // with_processing_units_count property implementation for execution
    // policies that simply forwards to the embedded executor (if that supports
    // the parameters type)

    HPX_CXX_EXPORT template <execution_policy ExPolicy>
        requires(std::invocable<with_processing_units_count_t,
            typename std::decay_t<ExPolicy>::executor_type, std::size_t>)
    constexpr decltype(auto) tag_invoke(
        with_processing_units_count_t, ExPolicy&& policy, std::size_t num_cores)
    {
        auto exec = with_processing_units_count(policy.executor(), num_cores);

        return create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    HPX_CXX_EXPORT template <execution_policy ExPolicy,
        executor_parameters Params>
        requires(
            std::invocable<with_processing_units_count_t,
                typename std::decay_t<ExPolicy>::executor_type, std::size_t> &&
            std::invocable<processing_units_count_t, std::decay_t<Params>,
                typename std::decay_t<ExPolicy>::executor_type,
                hpx::chrono::steady_duration const&, std::size_t>)
    constexpr decltype(auto) tag_invoke(
        with_processing_units_count_t, ExPolicy&& policy, Params&& params)
    {
        // explicitly extract pu count from given parameters object as otherwise
        // the executor might take precedence
        auto exec = with_processing_units_count(policy.executor(),
            processing_units_count(
                params, policy.executor(), hpx::chrono::null_duration, 0));

        return create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    // general fallback for parameters types that are not directly supported by
    // the underlying executor
    HPX_CXX_EXPORT template <typename ParametersProperty,
        execution_policy ExPolicy, executor_parameters Params>
    constexpr decltype(auto) tag_fallback_invoke(
        ParametersProperty, ExPolicy&& policy, Params&& params)
    {
        return policy.with(HPX_FORWARD(Params, params));
    }

    HPX_CXX_EXPORT template <typename ParametersProperty, typename ExPolicy,
        typename... Ts,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>)>
    constexpr auto tag_fallback_invoke(
        ParametersProperty prop, ExPolicy&& policy, Ts&&... ts)
        -> decltype(std::declval<ParametersProperty>()(
            std::declval<typename std::decay_t<ExPolicy>::executor_type>(),
            std::declval<Ts>()...))
    {
        return prop(policy.executor(), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::execution::experimental
