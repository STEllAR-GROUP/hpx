//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file find_here.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/id_type.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the global id representing this locality
    ///
    /// The function \a find_here() can be used to retrieve the global id
    /// usable to refer to the current locality.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     Generally, the id of a locality can be used for instance to
    ///           create new instances of components and to invoke plain actions
    ///           (global functions).
    ///
    /// \returns  The global id representing the locality this function has
    ///           been called on.
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
    HPX_EXPORT naming::id_type find_here(error_code& ec = throws);
}

