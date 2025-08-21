/// \file
/// Universal algorithm dispatch for Thrust integration with HPX

#pragma once

//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/thrust/detail/algorithm_map.hpp>
#include <hpx/thrust/policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/config/forward.hpp>   // HPX_FORWARD
#include <hpx/config/move.hpp>      // HPX_MOVE
#include <cuda_runtime.h>

#include <type_traits>

namespace hpx::thrust {

    template <typename HPXTag, typename ThrustPolicy, typename... Args,
        HPX_CONCEPT_REQUIRES_(
            is_thrust_execution_policy_v<std::decay_t<ThrustPolicy>>),
        typename = detail::is_algorithm_mapped<HPXTag, ThrustPolicy, Args...>>
    auto tag_invoke(HPXTag tag, ThrustPolicy&& policy, Args&&... args)
    {
        if constexpr (hpx::is_async_execution_policy_v<
                          std::decay_t<ThrustPolicy>>)
        {
            if constexpr (std::is_void_v<decltype(detail::algorithm_map<
                              HPXTag>::invoke(std::declval<ThrustPolicy>(),
                              std::declval<Args>()...))>)
            {
                detail::algorithm_map<HPXTag>::invoke(
                    HPX_FORWARD(ThrustPolicy, policy),
                    HPX_FORWARD(Args, args)...);
                return policy.get_future();    // future<void>
            }
            else
            {
                auto result = detail::algorithm_map<HPXTag>::invoke(
                    HPX_FORWARD(ThrustPolicy, policy),
                    HPX_FORWARD(Args, args)...);
                return policy.get_future().then(
                    [result = HPX_MOVE(result)](auto&& f) mutable {
                        f.get();
                        return HPX_MOVE(result);
                    });    // future<T>
            }
        }
        else
        {
            return detail::algorithm_map<HPXTag>::invoke(
                HPX_FORWARD(ThrustPolicy, policy), HPX_FORWARD(Args, args)...);
        }
    }
}    // namespace hpx::thrust
