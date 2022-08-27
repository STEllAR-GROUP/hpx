//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/traits/is_execution_policy.hpp

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx {
    namespace detail {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_execution_policy : std::false_type
        {
        };

        template <typename T>
        struct is_parallel_execution_policy : std::false_type
        {
        };

        template <typename T>
        struct is_sequenced_execution_policy : std::false_type
        {
        };

        template <typename T>
        struct is_async_execution_policy : std::false_type
        {
        };

        template <typename Executor>
        struct is_rebound_execution_policy : std::false_type
        {
        };

        template <typename Executor>
        struct is_vectorpack_execution_policy : std::false_type
        {
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// 1. The type is_execution_policy can be used to detect execution
    ///    policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_execution_policy<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_execution_policy is undefined.
    ///
    template <typename T>
    struct is_execution_policy
      : hpx::detail::is_execution_policy<typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy enables parallelization
    ///
    /// 1. The type is_parallel_execution_policy can be used to detect parallel
    ///    execution policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_parallel_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_parallel_execution_policy is undefined.
    ///
    template <typename T>
    struct is_parallel_execution_policy
      : hpx::detail::is_parallel_execution_policy<typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_parallel_execution_policy_v =
        is_parallel_execution_policy<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy does not enable
    ///            parallelization
    ///
    /// 1. The type is_sequenced_execution_policy can be used to detect
    ///    non-parallel execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_sequenced_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_sequenced_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_sequenced_execution_policy
      : hpx::detail::is_sequenced_execution_policy<typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_sequenced_execution_policy_v =
        is_sequenced_execution_policy<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy makes algorithms
    ///            asynchronous
    ///
    /// 1. The type is_async_execution_policy can be used to detect
    ///    asynchronous execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_async_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_async_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_async_execution_policy
      : hpx::detail::is_async_execution_policy<typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_async_execution_policy_v =
        is_async_execution_policy<T>::value;

    /// \cond NOINTERNAL
    template <typename T>
    struct is_rebound_execution_policy
      : hpx::detail::is_rebound_execution_policy<typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_rebound_execution_policy_v =
        is_rebound_execution_policy<T>::value;

    // extension:
    template <typename T>
    struct is_vectorpack_execution_policy
      : hpx::detail::is_vectorpack_execution_policy<
            typename std::decay<T>::type>
    {
    };

    template <typename T>
    inline constexpr bool is_vectorpack_execution_policy_v =
        is_vectorpack_execution_policy<T>::value;
    /// \endcond
}    // namespace hpx
