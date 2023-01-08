//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy_parameters.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/modules/concepts.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::parallel::execution {

    // with_processing_units_count property implementation for execution
    // policies that simply forwards to the embedded executor (if that supports
    // the parameters type)

    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::is_invocable_v<
                with_processing_units_count_t,
                typename std::decay_t<ExPolicy>::executor_type, std::size_t>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        with_processing_units_count_t, ExPolicy&& policy, std::size_t num_cores)
    {
        auto exec = with_processing_units_count(policy.executor(), num_cores);

        return create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    // clang-format off
    template <typename ExPolicy, typename Params,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_executor_parameters_v<Params> &&
            hpx::is_invocable_v<
                with_processing_units_count_t,
                typename std::decay_t<ExPolicy>::executor_type, std::size_t> &&
            detail::has_processing_units_count_v<std::decay_t<Params>>
        )>
    // clang-format on
    constexpr decltype(auto) tag_invoke(
        with_processing_units_count_t, ExPolicy&& policy, Params&& params)
    {
        // explicitly extract pu count from given parameters object as otherwise
        // the executor might take precedence
        auto exec = with_processing_units_count(policy.executor(),
            params.processing_units_count(
                policy.executor(), hpx::chrono::null_duration, 0));

        return create_rebound_policy(
            policy, HPX_MOVE(exec), policy.parameters());
    }

    // general fallback for parameters types that are not directly supported by
    // the underlying executor

    // clang-format off
    template <typename ParametersProperty, typename ExPolicy, typename Params,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_executor_parameters_v<Params>
        )>
    // clang-format on
    constexpr decltype(auto) tag_fallback_invoke(
        ParametersProperty, ExPolicy&& policy, Params&& params)
    {
        return policy.with(HPX_FORWARD(Params, params));
    }

    // clang-format off
    template <typename ParametersProperty, typename ExPolicy, typename...Ts,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    constexpr auto tag_fallback_invoke(
        ParametersProperty prop, ExPolicy&& policy, Ts&&... ts)
        -> decltype(std::declval<ParametersProperty>()(
            std::declval<typename std::decay_t<ExPolicy>::executor_type>(),
            std::declval<Ts>()...))
    {
        return prop(policy.executor(), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::parallel::execution
