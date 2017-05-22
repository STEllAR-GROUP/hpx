//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_THIS_THREAD_EXECUTORS_JUL_16_2015_0809PM)
#define HPX_PARALLEL_EXECUTORS_THIS_THREAD_EXECUTORS_JUL_16_2015_0809PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/parallel/executors/thread_execution_information.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>
#include <hpx/runtime/threads/executors/this_thread_executors.hpp>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using this_thread_static_queue_executor =
            threads::executors::this_thread_static_queue_executor;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using this_thread_static_priority_queue_executor =
            threads::executors::this_thread_static_priority_queue_executor;
#endif
}}}

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>

///////////////////////////////////////////////////////////////////////////////
// Compatibility layer
namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using this_thread_static_queue_executor =
        execution::this_thread_static_queue_executor;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using this_thread_static_priority_queue_executor =
        execution::this_thread_static_priority_queue_executor;
#endif
}}}
#endif

#endif
