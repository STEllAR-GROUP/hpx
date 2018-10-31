//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/executors/thread_execution.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>
#include <hpx/traits/executor_traits.hpp>

namespace hpx { namespace parallel { namespace execution
{
    /// A \a service_executor exposes one of the predefined HPX thread pools
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the given thread pool. The
    ///       tasks will not run as HPX-threads.
    ///
    using service_executor = threads::executors::service_executor;

    /// A \a io_pool_executor exposes the predefined HPX IO thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the IO thread pool. The
    ///       tasks will not run as HPX-threads.
    ///
    using io_pool_executor = threads::executors::io_pool_executor;

    /// A \a io_pool_executor exposes the predefined HPX parcel thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the parcel thread pool. The
    ///       tasks will not run as HPX-threads.
    ///
    using parcel_pool_executor = threads::executors::parcel_pool_executor;

    /// A \a io_pool_executor exposes the predefined HPX timer thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the timer thread pool. The
    ///       tasks will not run as HPX-threads.
    ///
    using timer_pool_executor = threads::executors::timer_pool_executor;

    /// A \a io_pool_executor exposes the predefined HPX main thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the main thread pool. The
    ///       tasks will not run as HPX-threads.
    ///
    using main_pool_executor = threads::executors::main_pool_executor;
}}}

#endif
