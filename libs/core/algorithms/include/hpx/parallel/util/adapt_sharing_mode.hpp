//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/executors/execution_policy_scheduling_property.hpp>

namespace hpx::parallel::util {

    // clang-format off
    template <typename ExPolicy,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy>
        )>
    // clang-format on
    decltype(auto) adapt_sharing_mode(
        ExPolicy&& policy, hpx::threads::thread_sharing_hint sharing)
    {
        static constexpr bool supports_sharing_hint =
            hpx::functional::is_tag_invocable_v<
                hpx::execution::experimental::with_hint_t,
                std::decay_t<ExPolicy>, hpx::threads::thread_schedule_hint>;

        if constexpr (supports_sharing_hint)
        {
            hpx::threads::thread_schedule_hint hint =
                hpx::execution::experimental::get_hint(policy);

            // modify sharing hint only if it is the default (do not overwrite
            // user supplied values)
            if (hint.sharing_mode() ==
                    hpx::threads::thread_sharing_hint::none &&
                sharing != hpx::threads::thread_sharing_hint::none)
            {
                hint.sharing_mode(sharing);
            }

            return hpx::execution::experimental::with_hint(
                HPX_FORWARD(ExPolicy, policy), hint);
        }
        else
        {
            return HPX_FORWARD(ExPolicy, policy);
        }
    }
}    // namespace hpx::parallel::util
