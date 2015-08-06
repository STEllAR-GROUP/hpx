//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/detail/thread_executor.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executors.hpp>
#include <hpx/util/move.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct local_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new local_queue_executor
        ///
        /// \param max_punits   [in] The maximum number of processing units to
        ///                     associate with the newly created executor.
        /// \param min_punits   [in] The minimum number of processing units to
        ///                     associate with the newly created executor
        ///                     (default: 1).
        ///
        explicit local_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(threads::executors::local_queue_executor(
                max_punits, min_punits))
        {}
    };
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct static_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new static_queue_executor
        ///
        /// \param max_punits   [in] The maximum number of processing units to
        ///                     associate with the newly created executor.
        /// \param min_punits   [in] The minimum number of processing units to
        ///                     associate with the newly created executor
        ///                     (default: 1).
        ///
        explicit static_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(threads::executors::static_queue_executor(
                max_punits, min_punits))
        {}
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct local_priority_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new local_priority_queue_executor
        ///
        /// \param max_punits   [in] The maximum number of processing units to
        ///                     associate with the newly created executor.
        /// \param min_punits   [in] The minimum number of processing units to
        ///                     associate with the newly created executor
        ///                     (default: 1).
        ///
        explicit local_priority_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(threads::executors::local_priority_queue_executor(
                max_punits, min_punits))
        {}
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct static_priority_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new static_priority_queue_executor
        ///
        /// \param max_punits   [in] The maximum number of processing units to
        ///                     associate with the newly created executor.
        /// \param min_punits   [in] The minimum number of processing units to
        ///                     associate with the newly created executor
        ///                     (default: 1).
        ///
        explicit static_priority_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(threads::executors::static_priority_queue_executor(
                max_punits, min_punits))
        {}
    };
#endif
}}}

#endif
