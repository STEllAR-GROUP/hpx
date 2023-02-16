//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/std_execution_policy.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_STD_EXECUTION_POLICES)
#include <hpx/execution/traits/is_execution_policy.hpp>

#include <execution>
#include <type_traits>

namespace hpx::detail {

    // Specialize our is_execution_policy traits for the corresponding std
    // versions

    /// \cond NOINTERNAL
    template <>
    struct is_execution_policy<std::execution::sequenced_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<std::execution::parallel_policy> : std::true_type
    {
    };

    template <>
    struct is_execution_policy<std::execution::parallel_unsequenced_policy>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<std::execution::parallel_policy>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<
        std::execution::parallel_unsequenced_policy> : std::true_type
    {
    };

    template <>
    struct is_sequenced_execution_policy<std::execution::sequenced_policy>
      : std::true_type
    {
    };

#if defined(HPX_HAVE_CXX20_STD_EXECUTION_POLICES)
    template <>
    struct is_execution_policy<std::execution::unsequenced_policy>
      : std::true_type
    {
    };

    template <>
    struct is_sequenced_execution_policy<std::execution::unsequenced_policy>
      : std::true_type
    {
    };
#endif
    /// \endcond
}    // namespace hpx::detail

#endif
