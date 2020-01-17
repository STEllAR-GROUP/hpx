//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP
#define HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/execution/executors/service_executors.hpp>

namespace hpx { namespace threads { namespace executors {
    using parallel::execution::service_executor_type;

    using parallel::execution::io_pool_executor;
    using parallel::execution::main_pool_executor;
    using parallel::execution::parcel_pool_executor;
    using parallel::execution::service_executor;
    using parallel::execution::timer_pool_executor;
}}}    // namespace hpx::threads::executors

#endif
#endif /* HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP */
