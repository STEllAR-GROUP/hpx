//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_fwd.hpp

#ifndef HPX_RUNTIME_FWD_HPP
#define HPX_RUNTIME_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/util_fwd.hpp>
#include <hpx/util/function.hpp>
#include <hpx/runtime/naming_fwd.hpp>

namespace hpx
{
    class HPX_API_EXPORT runtime;
    class HPX_API_EXPORT thread;

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    class HPX_API_EXPORT runtime_impl;

    /// The function \a get_runtime returns a reference to the (thread
    /// specific) runtime instance.
    HPX_API_EXPORT runtime& get_runtime();
    HPX_API_EXPORT runtime* get_runtime_ptr();

    /// The function \a get_locality returns a reference to the locality prefix
    HPX_API_EXPORT naming::gid_type const& get_locality();

    /// The function \a get_runtime_instance_number returns a unique number
    /// associated with the runtime instance the current thread is running in.
    HPX_API_EXPORT std::size_t get_runtime_instance_number();

    HPX_API_EXPORT void report_error(std::size_t num_thread
      , boost::exception_ptr const& e);

    HPX_API_EXPORT void report_error(boost::exception_ptr const& e);

    /// Register a function to be called during system shutdown
    HPX_API_EXPORT bool register_on_exit(util::function_nonser<void()> const&);

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT bool is_scheduler_numa_sensitive();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT util::runtime_configuration const& get_config();

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT hpx::util::io_service_pool* get_thread_pool(
        char const* name, char const* pool_name_suffix = "");

    /// \endcond
}

#endif
