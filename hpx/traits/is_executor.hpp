//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTOR_DEC_23_0759PM)
#define HPX_TRAITS_IS_EXECUTOR_DEC_23_0759PM

#include <hpx/config.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/traits/is_executor_v1.hpp>    // backwards compatibility

#include <type_traits>

namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        template <typename T>
        struct is_one_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_host_based_one_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_non_blocking_one_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_bulk_one_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_two_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_non_blocking_two_way_executor
          : std::false_type
        {};

        template <typename T>
        struct is_bulk_two_way_executor
          : std::false_type
        {};
    }

    // Executor type traits:

    // Condition: T meets the syntactic requirements for OneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_one_way_executor
      : detail::is_one_way_executor<typename std::decay<T>::type>
    {};

    // Condition: T meets the syntactic requirements for HostBasedOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_host_based_one_way_executor
      : detail::is_host_based_one_way_executor<typename std::decay<T>::type>
    {};

    // Condition: T meets the syntactic requirements for NonBlockingOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_non_blocking_one_way_executor
      : detail::is_non_blocking_one_way_executor<typename std::decay<T>::type>
    {};

    // Condition: T meets the syntactic requirements for BulkOneWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : detail::is_bulk_one_way_executor<typename std::decay<T>::type>
    {};

    // Condition: T meets the syntactic requirements for TwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_two_way_executor
      : detail::is_two_way_executor<typename std::decay<T>::type>
    {};

    template <typename T, typename Enable = void>
    struct is_non_blocking_two_way_executor
      : detail::is_non_blocking_two_way_executor<typename std::decay<T>::type>
    {};

    // Condition: T meets the syntactic requirements for BulkTwoWayExecutor
    // Precondition: T is a complete type
    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : detail::is_bulk_two_way_executor<typename std::decay<T>::type>
    {};

//     template <typename T>
//     constexpr bool is_one_way_executor_v =
//         is_one_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_host_based_one_way_executor_v =
//         is_host_based_one_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_non_blocking_one_way_executor_v =
//         is_non_blocking_one_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_bulk_one_way_executor_v =
//         is_bulk_one_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_two_way_executor_v =
//         is_two_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_non_blocking_two_way_executor_v =
//         is_non_blocking_two_way_executor<T>::value;
//
//     template <typename T>
//     constexpr bool is_bulk_two_way_executor_v =
//         is_bulk_two_way_executor<T>::value;
}}}

namespace hpx { namespace traits
{
    // Concurrency TS V2: executor framework
    template <typename T, typename Enable = void>
    struct is_one_way_executor
      : parallel::execution::is_one_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_host_based_one_way_executor
      : parallel::execution::is_host_based_one_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_non_blocking_one_way_executor
      : parallel::execution::is_non_blocking_one_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_bulk_one_way_executor
      : parallel::execution::is_bulk_one_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_two_way_executor
      : parallel::execution::is_two_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_non_blocking_two_way_executor
      : parallel::execution::is_non_blocking_two_way_executor<T>
    {};

    template <typename T, typename Enable = void>
    struct is_bulk_two_way_executor
      : parallel::execution::is_bulk_two_way_executor<T>
    {};

    // trait testing for any of the above
    template <typename T, typename Enable = void>
    struct is_executor_any
      : util::detail::any_of<
            is_one_way_executor<T>,
            is_host_based_one_way_executor<T>,
            is_non_blocking_one_way_executor<T>,
            is_bulk_one_way_executor<T>,
            is_two_way_executor<T>,
            is_non_blocking_two_way_executor<T>,
            is_bulk_two_way_executor<T>
        >
    {};
}}

#endif

