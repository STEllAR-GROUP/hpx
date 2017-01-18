//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/default_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM)
#define HPX_PARALLEL_EXECUTORS_DEFAULT_EXECUTOR_AUG_24_2015_0624PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/parallel/executors/thread_execution_information.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>
#include <hpx/runtime/threads/executors/default_executor.hpp>

namespace hpx { namespace parallel { namespace execution
{
    /// Refers to the currently used base-executor
    using default_executor = threads::executors::default_executor;
}}}

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    using default_executor = execution::default_executor;
}}}
#endif

#endif
