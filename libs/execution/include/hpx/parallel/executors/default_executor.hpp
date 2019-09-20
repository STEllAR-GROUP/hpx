//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/default_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM)
#define HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/execution_parameters.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/parallel/executors/thread_execution_information.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
    /// Refers to the currently used base-executor
    using default_executor = threads::executors::default_executor;
}}}    // namespace hpx::parallel::execution

#endif
