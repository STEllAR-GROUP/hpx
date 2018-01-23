//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx_finalize.hpp

#ifndef HPX_HPX_SUSPEND_HPP
#define HPX_HPX_SUSPEND_HPP

#include <hpx/exception_fwd.hpp>

/// \namespace hpx
namespace hpx
{
    /// \brief Main function to gracefully terminate the HPX runtime system.
    ///
    /// The function \a hpx::finalize is the main way to (gracefully) exit any
    /// HPX application. It should be called from one locality only (usually
    /// the console) and it will notify all connected localities to finish
    /// execution. Only after all other localities have exited this function
    /// will return, allowing to exit the console locality as well.
    ///
    /// During the execution of this function the runtime system will invoke
    /// all registered shutdown functions (see \a hpx::init) on all localities.
    ///
    /// \param shutdown_timeout This parameter allows to specify a timeout (in
    ///           microseconds), specifying how long any of the connected
    ///           localities should wait for pending tasks to be executed.
    ///           After this timeout, all suspended HPX-threads will be aborted.
    ///           Note, that this function will not abort any running
    ///           HPX-threads. In any case the shutdown will not proceed as long
    ///           as there is at least one pending/running HPX-thread.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           timeout value (can be set as the configuration parameter
    ///           `hpx.shutdown_timeout`), and if that is not set or `-1.0` as
    ///           well, it will disable any timeout, each connected
    ///           locality will wait for all existing HPX-threads to terminate.
    ///
    /// \param localwait This parameter allows to specify a local wait time
    ///           (in microseconds) before the connected localities will be
    ///           notified and the overall shutdown process starts.
    ///
    ///           The default value (`-1.0`) will try to find a globally set
    ///           wait time value (can be set as the configuration parameter
    ///           "hpx.finalize_wait_time"), and if this is not set or `-1.0`
    ///           as well, it will disable any addition local wait time before
    ///           proceeding.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  This function will always return zero.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// This function will block and wait for all connected localities to exit
    /// before returning to the caller. It should be the last HPX-function
    /// called by any application.
    ///
    /// Using this function is an alternative to \a hpx::disconnect, these
    /// functions do not need to be called both.
    HPX_EXPORT int suspend(error_code& ec = throws);

    /// \brief Main function to gracefully terminate the HPX runtime system.
    ///
    /// The function \a hpx::finalize is the main way to (gracefully) exit any
    /// HPX application. It should be called from one locality only (usually
    /// the console) and it will notify all connected localities to finish
    /// execution. Only after all other localities have exited this function
    /// will return, allowing to exit the console locality as well.
    ///
    /// During the execution of this function the runtime system will invoke
    /// all registered shutdown functions (see \a hpx::init) on all localities.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  This function will always return zero.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// This function will block and wait for all connected localities to exit
    /// before returning to the caller. It should be the last HPX-function
    /// called by any application.
    ///
    /// Using this function is an alternative to \a hpx::disconnect, these
    /// functions do not need to be called both.
    HPX_EXPORT int resume(error_code& ec = throws);
}

#endif /*HPX_HPX_SUSPEND_HPP*/
