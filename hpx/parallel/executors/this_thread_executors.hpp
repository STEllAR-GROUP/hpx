//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_pool_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THIS_THREAD_EXECUTORS_JUL_16_2015_0809PM)
#define HPX_PARALLEL_EXECUTORS_THIS_THREAD_EXECUTORS_JUL_16_2015_0809PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/detail/thread_executor.hpp>
#include <hpx/runtime/threads/executors/this_thread_executors.hpp>
#include <hpx/util/move.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct this_thread_local_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new this_thread_local_queue_executor
        ///
        explicit this_thread_local_queue_executor()
          : threads_executor(
                threads::executors::this_thread_local_queue_executor())
        {}
    };
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct this_thread_static_priority_queue_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        /// Creates a new this_thread_static_priority_queue_executor
        ///
        explicit this_thread_static_priority_queue_executor()
          : threads_executor(
                threads::executors::this_thread_static_priority_queue_executor())
        {}
    };
#endif

    namespace detail
    {
        /// \cond NOINTERNAL
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
        template <>
        struct is_executor<this_thread_local_queue_executor>
          : std::true_type
        {};
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
        template <>
        struct is_executor<this_thread_static_priority_queue_executor>
          : std::true_type
        {};
#endif
        /// \endcond
    }
}}}

#endif
