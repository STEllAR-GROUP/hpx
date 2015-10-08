//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_locality_id.hpp

#ifndef HPX_RUNTIME_GET_LOCALITY_ID_HPP
#define HPX_RUNTIME_GET_LOCALITY_ID_HPP

#include <hpx/config/export_definitions.hpp>

#include <hpx/exception_fwd.hpp>

#include <boost/cstdint.hpp>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the number of the locality this function is being called
    ///        from.
    ///
    /// This function returns the id of the current locality.
    ///
    /// \param ec [in,out] this represents the error status on exit, if this
    ///           is pre-initialized to \a hpx#throws the function will throw
    ///           on error instead.
    ///
    /// \note     The returned value is zero based and its maximum value is
    ///           smaller than the overall number of localities the current
    ///           application is running on (as returned by
    ///           \a get_num_localities()).
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           hpx::exception.
    ///
    /// \note     This function needs to be executed on a HPX-thread. It will
    ///           fail otherwise (it will return -1).
    HPX_API_EXPORT boost::uint32_t get_locality_id(error_code& ec = throws);
}

#endif
