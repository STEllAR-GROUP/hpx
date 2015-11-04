//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/service_executors.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM)
#define HPX_PARALLEL_EXECUTORS_SERVICE_EXECUTORS_MAY_15_2015_0548PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/thread_executor_traits.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>

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
        // Associate the static_chunk_size executor parameters type as a default
        // with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new service executor for the given HPX thread pool
        ///
        /// \param t    [in] Specifies the HPX thread pool to encapsulate
        /// \param name_suffix  [in] The name suffix to use for the underlying
        ///             thread pool
        ///
        service_executor(
                BOOST_SCOPED_ENUM(threads::executors::service_executor_type) t,
                char const* name_suffix = "")
          : threads::executors::service_executor(t, name_suffix)
        {}
    };
}}}

#endif
