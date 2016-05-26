//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file startup_function.hpp

#ifndef HPX_RUNTIME_SHUTDOWN_FUNCTION_HPP
#define HPX_RUNTIME_SHUTDOWN_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/util/unique_function.hpp>

namespace hpx
{
    /// The type of a function which is registered to be executed as a
    /// shutdown or pre-shutdown function.
    typedef util::unique_function_nonser<void()> shutdown_function_type;

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed before any shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_pre_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() before any of the registered shutdown functions are
    /// executed (see: \a hpx::register_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-shutdown function.
    ///
    /// \note If this function is called while the pre-shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_shutdown_function()
    HPX_API_EXPORT void register_pre_shutdown_function(shutdown_function_type f);

    /// \brief Add a function to be executed by a HPX thread during
    /// \a hpx::finalize() but guaranteed after any pre-shutdown function is
    /// executed (system-wide)
    ///
    /// Any of the functions registered with \a register_shutdown_function
    /// are guaranteed to be executed by an HPX thread during the execution of
    /// \a hpx::finalize() after any of the registered pre-shutdown functions
    /// are executed (see: \a hpx::register_pre_shutdown_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a shutdown function.
    ///
    /// \note If this function is called while the shutdown functions are
    ///       being executed, or after that point, it will raise a invalid_status
    ///       exception.
    ///
    /// \see    \a hpx::register_pre_shutdown_function()
    HPX_API_EXPORT void register_shutdown_function(shutdown_function_type f);
}

#endif
