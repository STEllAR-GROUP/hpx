//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_num_localities.hpp

#if !defined(HPX_RUNTIME_GET_NUM_LOCALITIES_HPP)
#define HPX_RUNTIME_GET_NUM_LOCALITIES_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/components/component_type.hpp>

#include <boost/cstdint.hpp>

namespace hpx
{
    /// \brief Return the number of localities which were registered at startup
    ///        for the running application.
    ///
    /// The function \a get_initial_num_localities returns the number of localities
    /// which were connected to the console at application startup.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_initial_num_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities asynchronously returns the
    /// number of localities currently connected to the console. The returned
    /// future represents the actual result.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities();

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_num_localities(launch::sync_policy,
        error_code& ec = throws);

    /// \brief Asynchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities asynchronously returns the
    /// number of localities currently connected to the console which support
    /// the creation of the given component type. The returned future represents
    /// the actual result.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT lcos::future<boost::uint32_t> get_num_localities(
        components::component_type t);

    /// \brief Synchronously return the number of localities which are
    ///        currently registered for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console which support the creation of the
    /// given component type. The returned future represents the actual result.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_API_EXPORT boost::uint32_t get_num_localities(launch::sync_policy,
        components::component_type t, error_code& ec = throws);

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This functions is deprecated, it will be removed in a future
    ///           version of HPX.
    ///
    /// \see      \a hpx::find_all_localities_sync, \a hpx::get_num_localities
    HPX_DEPRECATED(HPX_DEPRECATED_MSG)
    inline boost::uint32_t get_num_localities_sync(error_code& ec = throws)
    {
        return get_num_localities(launch::sync, ec);
    }

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    ///
    /// The function \a get_num_localities returns the number of localities
    /// currently connected to the console which support the creation of the
    /// given component type.
    ///
    /// \param t  The component type for which the number of connected
    ///           localities should be retrieved.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return 0 otherwise.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This functions is deprecated, it will be removed in a future
    ///           version of HPX.
    ///
    /// \see      \a hpx::find_all_localities, \a hpx::get_num_localities
    HPX_DEPRECATED(HPX_DEPRECATED_MSG)
    inline boost::uint32_t get_num_localities_sync(
        components::component_type t, error_code& ec = throws)
    {
        return get_num_localities(launch::sync, t, ec);
    }
#endif
}

#endif
