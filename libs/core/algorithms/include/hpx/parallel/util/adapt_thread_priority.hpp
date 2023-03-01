//  Copyright (c) 2023 Hartmut Kaiser
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
    decltype(auto) adapt_thread_priority(
        ExPolicy&& policy, hpx::threads::thread_priority new_priority)
    {
        static constexpr bool supports_priority =
            hpx::functional::is_tag_invocable_v<
                hpx::execution::experimental::with_priority_t,
                std::decay_t<ExPolicy>, hpx::threads::thread_priority>;

        if constexpr (supports_priority)
        {
            hpx::threads::thread_priority priority =
                hpx::execution::experimental::get_priority(policy);

            // modify priority only if it is the default (do not overwrite user
            // supplied values)
            if (priority == hpx::threads::thread_priority::default_ &&
                new_priority != hpx::threads::thread_priority::default_)
            {
                priority = new_priority;
            }

            return hpx::execution::experimental::with_priority(
                HPX_FORWARD(ExPolicy, policy), priority);
        }
        else
        {
            return HPX_FORWARD(ExPolicy, policy);
        }
    }
}    // namespace hpx::parallel::util
