//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file find_localities.hpp

#if !defined(HPX_RUNTIME_FIND_LOCALITIES_HPP)
#define HPX_RUNTIME_FIND_LOCALITIES_HPP

#include <hpx/config.hpp>
#include <hpx/errors.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>

#include <vector>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing the root locality
    ///
    /// The function \a find_root_locality() can be used to retrieve the global
    /// id usable to refer to the root locality. The root locality is the
    /// locality where the main AGAS service is hosted.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the root locality for this
    ///           application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_all_localities(), \a hpx::find_locality()
    HPX_API_EXPORT naming::id_type find_root_locality(error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the list of global ids representing all localities
    ///        available to this application which support the given component
    ///        type.
    ///
    /// The function \a find_all_localities() can be used to retrieve the
    /// global ids of all localities currently available to this application
    /// which support the creation of instances of the given component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available localities.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global ids representing the localities currently
    ///           available to this application which support the creation of
    ///           instances of the given component type. If no localities
    ///           supporting the given component type are currently available,
    ///           this function will return an empty vector.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_all_localities(
        components::component_type type, error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one).
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        error_code& ec = throws);

    /// \brief Return the list of locality ids of remote localities supporting
    ///        the given component type. By default this function will return
    ///        the list of all remote localities (all but the current locality).
    ///
    /// The function \a find_remote_localities() can be used to retrieve the
    /// global ids of all remote localities currently available to this
    /// application (i.e. all localities except the current one) which
    /// support the creation of instances of the given component type.
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return the available remote localities.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global ids representing the remote localities currently
    ///           available to this application.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return an empty vector otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_locality()
    HPX_API_EXPORT std::vector<naming::id_type> find_remote_localities(
        components::component_type type, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing an arbitrary locality which
    ///        supports the given component type.
    ///
    /// The function \a find_locality() can be used to retrieve the
    /// global id of an arbitrary locality currently available to this
    /// application which supports the creation of instances of the given
    /// component type.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \param type  [in] The type of the components for which the function should
    ///           return any available locality.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \returns  The global id representing an arbitrary locality currently
    ///           available to this application which supports the creation of
    ///           instances of the given component type. If no locality
    ///           supporting the given component type is currently available,
    ///           this function will return \a hpx::naming::invalid_id.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function will return meaningful results only if called
    ///           from an HPX-thread. It will return \a hpx::naming::invalid_id
    ///           otherwise.
    ///
    /// \see      \a hpx::find_here(), \a hpx::find_all_localities()
    HPX_API_EXPORT naming::id_type find_locality(components::component_type type,
        error_code& ec = throws);
}

#endif
