/// \file
/// Universal algorithm dispatch for Thrust integration with HPX

#pragma once

#include <hpx/async_cuda/thrust/detail/algorithm_map.hpp>
#include <hpx/async_cuda/thrust/policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

namespace hpx { namespace async_cuda { namespace thrust {

    template <typename HPXTag, typename ThrustPolicy, typename... Args,
        HPX_CONCEPT_REQUIRES_(
            is_thrust_execution_policy_v<std::decay_t<ThrustPolicy>>),
        typename = detail::is_algorithm_mapped<HPXTag, ThrustPolicy, Args...>>
    auto tag_invoke(HPXTag tag, ThrustPolicy&& policy, Args&&... args)
    {
        if constexpr (hpx::detail::is_async_execution_policy<
                          std::decay_t<ThrustPolicy>>::value)
        {
            if constexpr (std::is_void_v<decltype(detail::algorithm_map<
                              HPXTag>::invoke(std::declval<ThrustPolicy>(),
                              std::declval<Args>()...))>)
            {
                detail::algorithm_map<HPXTag>::invoke(
                    policy, std::forward<Args>(args)...);
                return policy.get_future();    // future<void>
            }
            else
            {
                auto result = detail::algorithm_map<HPXTag>::invoke(
                    policy, std::forward<Args>(args)...);
                return policy.get_future().then([result](auto&&) mutable {
                    return result;
                });    // future<T>
            }
        }
        else
        {
            return detail::algorithm_map<HPXTag>::invoke(
                policy, std::forward<Args>(args)...);
        }
    }
}}}    // namespace hpx::async_cuda::thrust
