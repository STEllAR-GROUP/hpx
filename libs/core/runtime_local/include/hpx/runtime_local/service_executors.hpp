//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/service_executors.hpp>

#include <cstdint>
#include <type_traits>

namespace hpx::parallel::execution {

    enum class service_executor_type : std::uint8_t
    {
        io_thread_pool,        ///< Selects creating a service executor using
                               ///< the I/O pool of threads
        parcel_thread_pool,    ///< Selects creating a service executor using
                               ///< the parcel pool of threads
        timer_thread_pool,     ///< Selects creating a service executor using
                               ///< the timer pool of threads
        main_thread            ///< Selects creating a service executor using
                               ///< the main thread
    };

    namespace detail {

        HPX_CORE_EXPORT hpx::util::io_service_pool* get_service_pool(
            service_executor_type t, char const* name_suffix = "");
    }    // namespace detail

    struct HPX_CORE_EXPORT service_executor : public detail::service_executor
    {
        explicit service_executor(
            service_executor_type t, char const* name_suffix = "");
    };

    struct HPX_CORE_EXPORT io_pool_executor : detail::service_executor
    {
        io_pool_executor();
    };

    struct HPX_CORE_EXPORT parcel_pool_executor : detail::service_executor
    {
        explicit parcel_pool_executor(char const* name_suffix = "-tcp");
    };

    struct HPX_CORE_EXPORT timer_pool_executor : detail::service_executor
    {
        timer_pool_executor();
    };

    struct HPX_CORE_EXPORT main_pool_executor : detail::service_executor
    {
        main_pool_executor();
    };

    ///  \cond NOINTERNAL
    template <>
    struct is_one_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<detail::service_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<service_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<io_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<parcel_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<timer_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_one_way_executor<main_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<main_pool_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<main_pool_executor> : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
