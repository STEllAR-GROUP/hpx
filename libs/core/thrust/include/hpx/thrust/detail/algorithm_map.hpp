/// \file
/// Template specialization mapping from HPX algorithm tags to Thrust functions
/// This file contains all the HPX tag -> Thrust function mappings using 

//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
//

#pragma once

// HPX algorithm headers for tag types
#include <hpx/parallel/algorithms/fill.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/algorithms/find.hpp>
#include <hpx/parallel/algorithms/count.hpp>

// Centralized Thrust algorithm headers
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/find.h>
#include <thrust/count.h>
#include <thrust/unique.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <hpx/config/forward.hpp>   // HPX_FORWARD
#include <type_traits>
#include <utility>

namespace hpx::thrust::detail {

template<typename HPXTag>
struct algorithm_map; // No definition = compilation error for unmapped algorithms

template<>
struct algorithm_map<hpx::fill_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::fill(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::fill"; }
};

template<>
struct algorithm_map<hpx::fill_n_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::fill_n(policy.get(), HPX_FORWARD(Args, args)...);
    }
};

template<>
struct algorithm_map<hpx::copy_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::copy(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::copy"; }
};

template<>
struct algorithm_map<hpx::transform_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::transform(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::transform"; }
};

template<>
struct algorithm_map<hpx::for_each_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::for_each(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::for_each"; }
};

template<>
struct algorithm_map<hpx::reduce_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::reduce(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::reduce"; }
};

template<>
struct algorithm_map<hpx::sort_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::sort(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::sort"; }
};

template<>
struct algorithm_map<hpx::find_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::find(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::find"; }
};

template<>
struct algorithm_map<hpx::count_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::count(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::count"; }
};


template<>
struct algorithm_map<hpx::unique_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::unique(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::unique"; }
};

template<>
struct algorithm_map<hpx::reverse_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::reverse(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::reverse"; }
};

template<>
struct algorithm_map<hpx::inclusive_scan_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::inclusive_scan(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::inclusive_scan"; }
};

template<>
struct algorithm_map<hpx::exclusive_scan_t> {
    template<typename Policy, typename... Args>
    static constexpr decltype(auto) invoke(Policy&& policy, Args&&... args) {
        return ::thrust::exclusive_scan(policy.get(), HPX_FORWARD(Args, args)...);
    }
    
    static constexpr char const* name() { return "thrust::exclusive_scan"; }
};


// SFINAE HELPER - Check if algorithm is mapped at compile time
// This is used in the universal tag_invoke to enable/disable the overload
template<typename HPXTag, typename Policy, typename... Args>
using is_algorithm_mapped = std::void_t<
    decltype(algorithm_map<HPXTag>::invoke(std::declval<Policy>(), std::declval<Args>()...))
>;

} // namespace hpx::thrust::detail