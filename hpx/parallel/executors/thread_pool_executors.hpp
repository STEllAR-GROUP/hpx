//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/parallel/executors/thread_execution_information.hpp>
#include <hpx/parallel/executors/thread_timed_execution.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executors.hpp>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    /// Creates a new local_queue_executor
    ///
    /// \param max_punits   [in] The maximum number of processing units to
    ///                     associate with the newly created executor.
    /// \param min_punits   [in] The minimum number of processing units to
    ///                     associate with the newly created executor
    ///                     (default: 1).
    ///
    using local_queue_executor = threads::executors::local_queue_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    /// Creates a new static_queue_executor
    ///
    /// \param max_punits   [in] The maximum number of processing units to
    ///                     associate with the newly created executor.
    /// \param min_punits   [in] The minimum number of processing units to
    ///                     associate with the newly created executor
    ///                     (default: 1).
    ///
    using static_queue_executor = threads::executors::static_queue_executor;
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
    /// Creates a new throttle_queue_executor
    ///
    /// \param max_punits   [in] The maximum number of processing units to
    ///                     associate with the newly created executor.
    /// \param min_punits   [in] The minimum number of processing units to
    ///                     associate with the newly created executor
    ///                     (default: 1).
    ///
    using throttle_queue_executor = threads::executors::throttle_queue_executor;
#endif

    /// Creates a new local_priority_queue_executor
    ///
    /// \param max_punits   [in] The maximum number of processing units to
    ///                     associate with the newly created executor.
    /// \param min_punits   [in] The minimum number of processing units to
    ///                     associate with the newly created executor
    ///                     (default: 1).
    ///
    using local_priority_queue_executor =
        threads::executors::local_priority_queue_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    /// Creates a new static_priority_queue_executor
    ///
    /// \param max_punits   [in] The maximum number of processing units to
    ///                     associate with the newly created executor.
    /// \param min_punits   [in] The minimum number of processing units to
    ///                     associate with the newly created executor
    ///                     (default: 1).
    ///
    using static_priority_queue_executor =
        threads::executors::static_priority_queue_executor;
#endif
}}}

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/parallel/executors/thread_executor_traits.hpp>

///////////////////////////////////////////////////////////////////////////////
// Compatibility layer
namespace hpx { namespace parallel { inline namespace v3
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_executor =
        threads::executors::local_queue_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_executor =
        threads::executors::static_queue_executor;
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
    using throttle_queue_executor =
        threads::executors::throttle_queue_executor;
#endif

    using local_priority_queue_executor =
        threads::executors::local_priority_queue_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_executor =
        threads::executors::static_priority_queue_executor;
#endif
}}}
#endif

#endif
