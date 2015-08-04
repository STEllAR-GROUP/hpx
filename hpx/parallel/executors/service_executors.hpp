//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/executors/detail/thread_executor.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>
#include <hpx/util/move.hpp>

#include <type_traits>

#include <boost/detail/scoped_enum_emulation.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// The type of the HPX thread pool to use for a given service_executor
    ///
    /// This enum type allows to specify the kind of the HPX thread pool to use
    /// for a given \a service_executor.
    BOOST_SCOPED_ENUM_START(service_executor_type)
    {
        io_thread_pool,        ///< Selects creating a service executor using
                               ///< the I/O pool of threads
        parcel_thread_pool,    ///< Selects creating a service executor using
                               ///< the parcel pool of threads
        timer_thread_pool,     ///< Selects creating a service executor using
                               ///< the timer pool of threads
        main_thread            ///< Selects creating a service executor using
                               ///< the main thread
    };
    BOOST_SCOPED_ENUM_END

    namespace detail
    {
        /// \cond NOINTERNAL
        inline threads::executor
        get_service_executor(BOOST_SCOPED_ENUM(service_executor_type) t)
        {
            switch(t)
            {
            case service_executor_type::io_thread_pool:
                return threads::executors::io_pool_executor();

            case service_executor_type::parcel_thread_pool:
                return threads::executors::parcel_pool_executor();

            case service_executor_type::timer_thread_pool:
                return threads::executors::timer_pool_executor();

            case service_executor_type::main_thread:
                return threads::executors::main_pool_executor();

            default:
                break;
            }

            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::parallel::v3::detail::get_service_executor",
                "unknown pool executor type");
        }
        /// \endcond
    }

    /// A \a service_executor exposes one of the predefined HPX thread pools
    /// through an executor interface.
    ///
    /// \note All tasks executed by one of these executors will run on
    ///       one of the OS-threads dedicated for the given thread pool. The
    ///       tasks will not run a HPX-threads.
    ///
    struct service_executor
#if !defined(DOXYGEN)
      : detail::threads_executor
#endif
    {
        // Associate the static_chunk_size executor parameters type as a default
        // with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new service executor for the given HPX thread pool
        ///
        /// \param t    [in] Specifies the HPX thread pool to encapsulate
        ///
        service_executor(BOOST_SCOPED_ENUM(service_executor_type) t)
          : threads_executor(detail::get_service_executor(t))
        {}
    };
}}}

#endif
