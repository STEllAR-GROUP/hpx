//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    namespace detail {
        template <typename T>
        struct is_one_way_executor : std::false_type
        {
        };

        template <typename T>
        struct is_never_blocking_one_way_executor : std::false_type
        {
        };

        template <typename T>
        struct is_bulk_one_way_executor : std::false_type
        {
        };

        template <typename T>
        struct is_two_way_executor : std::false_type
        {
        };

        template <typename T>
        struct is_bulk_two_way_executor : std::false_type
        {
        };
    }    // namespace detail

    // Executor type traits:

    // Condition: T meets the syntactic requirements for OneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_one_way_executor
      : detail::is_one_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_one_way_executor_t = typename is_one_way_executor<T>::type;

    // Condition: T meets the syntactic requirements for NonBlockingOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_never_blocking_one_way_executor
      : detail::is_never_blocking_one_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_never_blocking_one_way_executor_t =
        typename is_never_blocking_one_way_executor<T>::type;

    // Condition: T meets the syntactic requirements for BulkOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : detail::is_bulk_one_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_bulk_one_way_executor_t =
        typename is_bulk_one_way_executor<T>::type;

    // Condition: T meets the syntactic requirements for TwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_two_way_executor
      : detail::is_two_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_two_way_executor_t = typename is_two_way_executor<T>::type;

    // Condition: T meets the syntactic requirements for BulkTwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : detail::is_bulk_two_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_bulk_two_way_executor_t =
        typename is_bulk_two_way_executor<T>::type;

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool is_one_way_executor_v = is_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_never_blocking_one_way_executor_v =
        is_never_blocking_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_bulk_one_way_executor_v =
        is_bulk_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_two_way_executor_v = is_two_way_executor<T>::value;

    template <typename T>
    constexpr bool is_bulk_two_way_executor_v =
        is_bulk_two_way_executor<T>::value;
#endif
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace traits {
    // Concurrency TS V2: executor framework
    template <typename T, typename Enable = void>
    struct is_one_way_executor
      : parallel::execution::is_one_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_one_way_executor_t = typename is_one_way_executor<T>::type;

    template <typename T, typename Enable = void>
    struct is_never_blocking_one_way_executor
      : parallel::execution::is_never_blocking_one_way_executor<
            typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_never_blocking_one_way_executor_t =
        typename is_never_blocking_one_way_executor<T>::type;

    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : parallel::execution::is_bulk_one_way_executor<
            typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_bulk_one_way_executor_t =
        typename is_bulk_one_way_executor<T>::type;

    template <typename T, typename Enable = void>
    struct is_two_way_executor
      : parallel::execution::is_two_way_executor<typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_two_way_executor_t = typename is_two_way_executor<T>::type;

    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : parallel::execution::is_bulk_two_way_executor<
            typename std::decay<T>::type>
    {
    };

    template <typename T>
    using is_bulk_two_way_executor_t =
        typename is_bulk_two_way_executor<T>::type;

    // trait testing for any of the above
    template <typename T, typename Enable = void>
    struct is_executor_any
      : util::any_of<is_one_way_executor<T>,
            is_never_blocking_one_way_executor<T>, is_bulk_one_way_executor<T>,
            is_two_way_executor<T>, is_bulk_two_way_executor<T>>
    {
    };

    template <typename T>
    using is_executor_any_t = typename is_executor_any<T>::type;

#if defined(HPX_HAVE_CXX17_VARIABLE_TEMPLATES)
    template <typename T>
    constexpr bool is_one_way_executor_v = is_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_never_blocking_one_way_executor_v =
        is_never_blocking_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_bulk_one_way_executor_v =
        is_bulk_one_way_executor<T>::value;

    template <typename T>
    constexpr bool is_two_way_executor_v = is_two_way_executor<T>::value;

    template <typename T>
    constexpr bool is_bulk_two_way_executor_v =
        is_bulk_two_way_executor<T>::value;

    template <typename T>
    constexpr bool is_executor_any_v = is_executor_any<T>::value;
#endif
}}    // namespace hpx::traits
