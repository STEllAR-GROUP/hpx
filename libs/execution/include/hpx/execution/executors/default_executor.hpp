//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/default_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM)
#define HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/thread_execution.hpp>
#include <hpx/execution/executors/thread_execution_information.hpp>
#include <hpx/execution/executors/thread_timed_execution.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
    using default_executor = parallel_executor;
}}}    // namespace hpx::parallel::execution

#endif
