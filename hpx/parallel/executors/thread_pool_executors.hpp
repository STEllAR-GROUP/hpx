//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_THREAD_POOL_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/thread_executor.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executors.hpp>
#include <hpx/util/move.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct local_queue_executor : threads_executor
    {
        explicit local_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(
                threads::local_queue_executor(max_punits, min_punits))
        {}
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct local_priority_queue_executor : threads_executor
    {
        explicit local_priority_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(
                threads::local_priority_queue_executor(max_punits, min_punits))
        {}
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct static_priority_queue_executor : threads_executor
    {
        explicit static_priority_queue_executor(std::size_t max_punits,
                std::size_t min_punits = 1)
          : threads_executor(
                threads::static_priority_queue_executor(max_punits, min_punits))
        {}
    };
#endif

    namespace detail
    {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
        template <>
        struct is_executor<local_queue_executor>
          : std::true_type
        {};
#endif

        template <>
        struct is_executor<local_priority_queue_executor>
          : std::true_type
        {};

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
        template <>
        struct is_executor<static_priority_queue_executor>
          : std::true_type
        {};
#endif
        /// \endcond
    }
}}}

#endif
