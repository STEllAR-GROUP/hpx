//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/pool_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_POOL_EXECUTOR_FEB_17_2018_0327PM)
#define HPX_PARALLEL_EXECUTORS_POOL_EXECUTOR_FEB_17_2018_0327PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_parameters.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/parallel/executors/thread_execution_information.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    using pool_executor = threads::executors::pool_executor;
}}}    // namespace hpx::parallel::execution

#endif
