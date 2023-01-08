//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>

namespace hpx::parallel::execution {

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

        template <typename T>
        struct is_scheduler_executor : std::false_type
        {
        };
    }    // namespace detail

    // Executor type traits:

    // Condition: T meets the syntactic requirements for OneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_one_way_executor : detail::is_one_way_executor<std::decay_t<T>>
    {
    };

    // Condition: T meets the syntactic requirements for NonBlockingOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_never_blocking_one_way_executor
      : detail::is_never_blocking_one_way_executor<std::decay_t<T>>
    {
    };

    // Condition: T meets the syntactic requirements for BulkOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : detail::is_bulk_one_way_executor<std::decay_t<T>>
    {
    };

    // Condition: T meets the syntactic requirements for TwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_two_way_executor : detail::is_two_way_executor<std::decay_t<T>>
    {
    };

    // Condition: T meets the syntactic requirements for BulkTwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : detail::is_bulk_two_way_executor<std::decay_t<T>>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // is_scheduler_executor evaluates to true for executors that return senders
    // from their scheduling functions
    template <typename T, typename Enable = void>
    struct is_scheduler_executor
      : detail::is_scheduler_executor<std::decay_t<T>>
    {
    };
}    // namespace hpx::parallel::execution

namespace hpx::traits {

    // Concurrency TS V2: executor framework
    template <typename T, typename Enable = void>
    struct is_one_way_executor
      : parallel::execution::is_one_way_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_one_way_executor_v = is_one_way_executor<T>::value;

    template <typename T, typename Enable = void>
    struct is_never_blocking_one_way_executor
      : parallel::execution::is_never_blocking_one_way_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_never_blocking_one_way_executor_v =
        is_never_blocking_one_way_executor<T>::value;

    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : parallel::execution::is_bulk_one_way_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_bulk_one_way_executor_v =
        is_bulk_one_way_executor<T>::value;

    template <typename T, typename Enable = void>
    struct is_two_way_executor
      : parallel::execution::is_two_way_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_two_way_executor_v = is_two_way_executor<T>::value;

    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : parallel::execution::is_bulk_two_way_executor<std::decay_t<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_bulk_two_way_executor_v =
        is_bulk_two_way_executor<T>::value;

    // trait testing for any of the above
    template <typename T, typename Enable = void>
    struct is_executor_any
      : util::any_of<is_one_way_executor<T>,
            is_never_blocking_one_way_executor<T>, is_bulk_one_way_executor<T>,
            is_two_way_executor<T>, is_bulk_two_way_executor<T>>
    {
    };

    template <typename T>
    inline constexpr bool is_executor_any_v = is_executor_any<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    // is_scheduler_executor evaluates to true for executors that return senders
    // from their scheduling functions
    template <typename T, typename Enable = void>
    struct is_scheduler_executor : parallel::execution::is_scheduler_executor<T>
    {
    };

    template <typename T>
    inline constexpr bool is_scheduler_executor_v =
        is_scheduler_executor<T>::value;
}    // namespace hpx::traits
