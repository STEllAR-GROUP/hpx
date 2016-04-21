//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_colocation_id.hpp

#if !defined(HPX_RUNTIME_GET_COLOCATION_ID_HPP)
#define HPX_RUNTIME_GET_COLOCATION_ID_HPP

#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/naming/id_type.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the id of the locality where the object referenced by the
    ///        given id is currently located on
    ///
    /// The function hpx::get_colocation_id() returns the id of the locality
    /// where the given object is currently located.
    ///
    /// \param id [in] The id of the object to locate.
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \see    \a hpx::get_colocation_id()
    HPX_API_EXPORT naming::id_type get_colocation_id_sync(
        naming::id_type const& id, error_code& ec = throws);

    /// \brief Asynchronously return the id of the locality where the object
    ///        referenced by the given id is currently located on
    ///
    /// \see    \a hpx::get_colocation_id_sync()
    HPX_API_EXPORT lcos::future<naming::id_type> get_colocation_id(
        naming::id_type const& id);
}

#endif
