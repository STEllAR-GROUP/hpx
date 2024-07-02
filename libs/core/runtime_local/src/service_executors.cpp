//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/io_service/io_service_pool_fwd.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/runtime_local/service_executors.hpp>

namespace hpx::parallel::execution {

    namespace detail {

        hpx::util::io_service_pool* get_service_pool(
            service_executor_type t, char const* name_suffix)
        {
            switch (t)
            {
            case service_executor_type::io_thread_pool:
                return get_thread_pool("io-pool");

            case service_executor_type::parcel_thread_pool:
            {
                char const* suffix =
                    (name_suffix && *name_suffix) ? name_suffix : "-tcp";
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
        }
    }    // namespace detail

    service_executor::service_executor(
        service_executor_type t, char const* name_suffix)
      : detail::service_executor(detail::get_service_pool(t, name_suffix))
    {
    }

    io_pool_executor::io_pool_executor()
      : detail::service_executor(
            detail::get_service_pool(service_executor_type::io_thread_pool))
    {
    }

    parcel_pool_executor::parcel_pool_executor(char const* name_suffix)
      : detail::service_executor(detail::get_service_pool(
            service_executor_type::parcel_thread_pool, name_suffix))
    {
    }

    timer_pool_executor::timer_pool_executor()
      : detail::service_executor(
            detail::get_service_pool(service_executor_type::timer_thread_pool))
    {
    }

    main_pool_executor::main_pool_executor()
      : detail::service_executor(
            detail::get_service_pool(service_executor_type::main_thread))
    {
    }
}    // namespace hpx::parallel::execution
