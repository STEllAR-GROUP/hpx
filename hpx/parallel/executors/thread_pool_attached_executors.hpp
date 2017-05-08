//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_AUG_28_2015_0511PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_AUG_28_2015_0511PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/runtime/threads/executors/thread_pool_attached_executors.hpp>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_attached_executor =
        threads::executors::local_queue_attached_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_attached_executor =
        threads::executors::static_queue_attached_executor;
#endif

    using local_priority_queue_attached_executor =
        threads::executors::local_priority_queue_attached_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_attached_executor =
        threads::executors::static_priority_queue_attached_executor;
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
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_attached_executor =
        execution::local_queue_attached_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_attached_executor =
        execution::static_queue_attached_executor;
#endif

    using local_priority_queue_attached_executor =
        execution::local_priority_queue_attached_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_attached_executor =
        execution::static_priority_queue_attached_executor;
#endif
}}}
#endif

#endif
