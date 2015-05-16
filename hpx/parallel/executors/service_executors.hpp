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
#include <hpx/parallel/executors/detail/thread_executor.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>
#include <hpx/util/move.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    struct io_pool_executor : detail::threads_executor
    {
        io_pool_executor()
          : threads_executor(threads::executors::io_pool_executor())
        {}
    };

    struct parcel_pool_executor : detail::threads_executor
    {
        parcel_pool_executor()
          : threads_executor(threads::executors::parcel_pool_executor())
        {}
    };

    struct timer_pool_executor : detail::threads_executor
    {
        explicit timer_pool_executor()
          : threads_executor(threads::executors::timer_pool_executor())
        {}
    };

    struct main_pool_executor : detail::threads_executor
    {
        explicit main_pool_executor()
          : threads_executor(threads::executors::main_pool_executor())
        {}
    };

    namespace detail
    {
        template <>
        struct is_executor<io_pool_executor>
          : std::true_type
        {};

        template <>
        struct is_executor<parcel_pool_executor>
          : std::true_type
        {};

        template <>
        struct is_executor<timer_pool_executor>
          : std::true_type
        {};

        template <>
        struct is_executor<main_pool_executor>
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
