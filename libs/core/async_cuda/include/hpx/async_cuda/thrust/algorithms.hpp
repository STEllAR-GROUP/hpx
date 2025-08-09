/// \file
/// Universal algorithm dispatch for Thrust integration with HPX

#pragma once

#include <hpx/async_cuda/thrust/policy.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/async_cuda/thrust/detail/algorithm_map.hpp>
#include <hpx/concepts/concepts.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

namespace hpx {
namespace async_cuda {
namespace thrust {

template<typename HPXTag, typename ThrustPolicy, typename... Args,
    HPX_CONCEPT_REQUIRES_(
        is_thrust_execution_policy_v<std::decay_t<ThrustPolicy>>
    )>
auto tag_invoke(HPXTag tag, ThrustPolicy&& policy, Args&&... args) 
    -> decltype(detail::algorithm_map<HPXTag>::invoke(std::forward<ThrustPolicy>(policy), std::forward<Args>(args)...)) {
    
    
    // Universal dispatch to the mapped Thrust function
    // This calls detail::algorithm_map<HPXTag>::invoke(policy, args...)
    // which in turn calls the appropriate ::thrust::algorithm(policy.get(), args...)
    return detail::algorithm_map<HPXTag>::invoke(
        std::forward<ThrustPolicy>(policy), 
        std::forward<Args>(args)...
    );
}
} // namespace thrust
} // namespace async_cuda
} // namespace hpx
