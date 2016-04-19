//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file startup_function.hpp

#ifndef HPX_RUNTIME_STARTUP_FUNCTION_HPP
#define HPX_RUNTIME_STARTUP_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/util/function.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// The type of a function which is registered to be executed as a
    /// startup or pre-startup function.
    typedef util::function_nonser<void()> startup_function_type;

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed before any startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_pre_startup_function
    /// are guaranteed to be executed by an HPX thread before any of the
    /// registered startup functions are executed (see
    /// \a hpx::register_startup_function()).
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a pre-startup function.
    ///
    /// \note If this function is called while the pre-startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_startup_function()
    HPX_API_EXPORT void register_pre_startup_function(startup_function_type const& f);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Add a function to be executed by a HPX thread before hpx_main
    /// but guaranteed after any pre-startup function is executed (system-wide).
    ///
    /// Any of the functions registered with \a register_startup_function
    /// are guaranteed to be executed by an HPX thread after any of the
    /// registered pre-startup functions are executed (see:
    /// \a hpx::register_pre_startup_function()), but before \a hpx_main is
    /// being called.
    ///
    /// \param f  [in] The function to be registered to run by an HPX thread as
    ///           a startup function.
    ///
    /// \note If this function is called while the startup functions are
    ///       being executed or after that point, it will raise a invalid_status
    ///       exception.
    ///
    ///       This function is one of the few API functions which can be called
    ///       before the runtime system has been fully initialized. It will
    ///       automatically stage the provided startup function to the runtime
    ///       system during its initialization (if necessary).
    ///
    /// \see    \a hpx::register_pre_startup_function()
    HPX_API_EXPORT void register_startup_function(startup_function_type const& f);
}

#endif
