//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executors.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EMBEDDED_THREAD_POOLS_COMPATIBILITY)
#include <hpx/thread_executors/embedded_thread_pool_executors.hpp>
#include <hpx/thread_executors/thread_execution.hpp>

namespace hpx { namespace parallel { namespace execution {
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
}}}    // namespace hpx::parallel::execution
#endif
