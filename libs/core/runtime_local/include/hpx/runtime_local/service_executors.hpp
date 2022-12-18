//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/service_executors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution {
    enum class service_executor_type
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
        inline hpx::util::io_service_pool* get_service_pool(
            service_executor_type t, char const* name_suffix = "")
        {
            switch (t)
            {
            case service_executor_type::io_thread_pool:
                return get_thread_pool("io-pool");

            case service_executor_type::parcel_thread_pool:
            {
                char const* suffix = *name_suffix ? name_suffix : "-tcp";
                return get_thread_pool("parcel-pool", suffix);
            }

            case service_executor_type::timer_thread_pool:
                return get_thread_pool("timer-pool");

            case service_executor_type::main_thread:
                return get_thread_pool("main-pool");

            default:
                break;
            }

            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "hpx::threads::detail::get_service_pool",
                "unknown pool executor type");
            return nullptr;
        }
    }    // namespace detail

    struct service_executor : public detail::service_executor
    {
        service_executor(service_executor_type t, char const* name_suffix = "")
          : detail::service_executor(detail::get_service_pool(t, name_suffix))
        {
        }
    };

    struct io_pool_executor : public detail::service_executor
    {
        io_pool_executor()
          : detail::service_executor(
                detail::get_service_pool(service_executor_type::io_thread_pool))
        {
        }
    };

    struct parcel_pool_executor : public detail::service_executor
    {
        parcel_pool_executor(char const* name_suffix = "-tcp")
          : detail::service_executor(detail::get_service_pool(
                service_executor_type::parcel_thread_pool, name_suffix))
        {
        }
    };

    struct timer_pool_executor : public detail::service_executor
    {
        timer_pool_executor()
          : detail::service_executor(detail::get_service_pool(
                service_executor_type::timer_thread_pool))
        {
        }
    };

    struct main_pool_executor : public detail::service_executor
    {
        main_pool_executor()
          : detail::service_executor(
                detail::get_service_pool(service_executor_type::main_thread))
        {
        }
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
}}}    // namespace hpx::parallel::execution
