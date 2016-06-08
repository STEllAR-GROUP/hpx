//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    /// A \a service_executor exposes one of the predefined HPX thread pools
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the given thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct service_executor
#if !defined(DOXYGEN)
      : threads::executors::service_executor
#endif
    {
        /// Associate the static_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new service executor for the given HPX thread pool
        ///
        /// \param t    [in] Specifies the HPX thread pool to encapsulate
        /// \param name_suffix  [in] The name suffix to use for the underlying
        ///             thread pool
        ///
        service_executor(
                threads::executors::service_executor_type t,
                char const* name_suffix = "")
          : threads::executors::service_executor(t, name_suffix)
        {}
    };

    /// A \a io_pool_executor exposes the predefined HPX IO thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the IO thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct io_pool_executor : service_executor
    {
        /// Create a new service executor for the IO thread pool
        io_pool_executor()
          : service_executor(
                threads::executors::service_executor_type::io_thread_pool)
        {}
    };

    /// A \a io_pool_executor exposes the predefined HPX parcel thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the parcel thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct parcel_pool_executor : service_executor
    {
        /// Create a new service executor for the parcel thread pool
        ///
        /// \param name_suffix  [in] The name suffix to use for the underlying
        ///                     thread pool (default: "-tcp")
        parcel_pool_executor(char const* name_suffix = "-tcp")
          : service_executor(
                threads::executors::service_executor_type::parcel_thread_pool,
                name_suffix)
        {}
    };

    /// A \a io_pool_executor exposes the predefined HPX timer thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the timer thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct timer_pool_executor : service_executor
    {
        /// Create a new service executor for the timer thread pool
        timer_pool_executor()
          : service_executor(
                threads::executors::service_executor_type::timer_thread_pool)
        {}
    };

    /// A \a io_pool_executor exposes the predefined HPX main thread pool
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the main thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct main_pool_executor : service_executor
    {
        /// Create a new service executor for the main thread pool
        main_pool_executor()
          : service_executor(
                threads::executors::service_executor_type::main_thread)
        {}
    };
}}}

#endif
