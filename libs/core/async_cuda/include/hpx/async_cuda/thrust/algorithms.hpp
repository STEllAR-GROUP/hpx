/// \file
/// Direct algorithm implementations for Thrust integration with HPX

#pragma once

#include <hpx/async_cuda/thrust/policy.hpp>
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/concepts/concepts.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>

// TODO: Add thrust headers when available
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// #include <thrust/for_each.h>
// #include <thrust/copy.h>

namespace hpx {
namespace async_cuda {
namespace thrust {

// Constrained tag_invoke for hpx::fill with thrust policy
// This overload gets called when user does: hpx::fill(thrust_policy, ...)
template <typename ExPolicy, typename FwdIter, typename T,
    HPX_CONCEPT_REQUIRES_(
        hpx::async_cuda::thrust::is_thrust_execution_policy<
            std::decay_t<ExPolicy>>::value
    )>
decltype(auto) tag_invoke(
    hpx::fill_t, 
    ExPolicy const& policy,
    FwdIter first, FwdIter last, T const& value)
{
    // Assert that this is indeed a thrust policy at compile-time
    static_assert(
        hpx::async_cuda::thrust::is_thrust_execution_policy<
            std::decay_t<ExPolicy>>::value,
        "tag_invoke(hpx::fill_t) can only be called with Thrust execution policies"
    );

    // Log which policy is being used (debugging)
    // std::cout << "\n LOGGING POLICY INFO ";
    // log_policy_info(policy, "hpx::fill");
    // std::cout << " LOGGING POLICY INFO \n";   
    // std::cout << "📋 Thrust policy dispatch working! Policy type: " 
    //           << typeid(std::decay_t<ExPolicy>).name() << std::endl;

    // Use the underlying thrust execution policy via .get()
    ::thrust::fill(policy.get(), first, last, value);

    // thrust_policy - the base policy 
    // thrust_host_policy: thrust_policy --> ::thrust::host
    // thrust_device_policy: thrust_policy --> ::thrust::device
    
    return first;
}

template <typename ExPolicy, typename FwdIter, typename F,
    HPX_CONCEPT_REQUIRES_(
        hpx::async_cuda::thrust::is_thrust_execution_policy<
            std::decay_t<ExPolicy>>::value
    )>
decltype(auto) tag_invoke(
    hpx::for_each_t,
    ExPolicy const& policy, 
    FwdIter first, FwdIter last, F&& f)
{
    // Assert that this is indeed a thrust policy at compile-time
    static_assert(
        hpx::async_cuda::thrust::is_thrust_execution_policy<
            std::decay_t<ExPolicy>>::value,
        "tag_invoke(hpx::for_each_t) can only be called with Thrust execution policies"
    );
    
    
    // For now, placeholder CPU implementation  
    for (auto it = first; it != last; ++it) {
        f(*it);
    }
    return first;
}

// TODO: Add more constrained algorithms as needed
// - thrust::copy
// - thrust::transform
// - thrust::reduce
// - etc.

}}} // namespace hpx::async_cuda::thrust



// // Helper function to identify and log thrust policy types
// template<typename ExPolicy>
// void log_policy_info(const ExPolicy& policy, const char* algorithm_name) {
//     std::cout << " [" << algorithm_name << "] ";
    
//     if constexpr (std::is_same_v<std::decay_t<ExPolicy>, thrust_host_policy>) {
//         std::cout << "HOST policy (thrust::host - CPU execution)" << std::endl;
//     } else if constexpr (std::is_same_v<std::decay_t<ExPolicy>, thrust_device_policy>) {
//         std::cout << "DEVICE policy (thrust::device - GPU execution)" << std::endl;
//     } else {
//         std::cout << "Policy type: " << typeid(std::decay_t<ExPolicy>).name() << std::endl;
//     }
// }