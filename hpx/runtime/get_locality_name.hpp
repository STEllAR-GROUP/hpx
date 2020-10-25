//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/get_locality_name.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <string>

namespace hpx
{
    /// \fn std::string get_locality_name()
    ///
    /// \brief Return the name of the locality this function is called on.
    ///
    /// This function returns the name for the locality on which this function
    /// is called.
    ///
    /// \returns  This function returns the name for the locality on which the
    ///           function is called. The name is retrieved from the underlying
    ///           networking layer and may be different for different parcelports.
    ///
    /// \see      \a future<std::string> get_locality_name(naming::id_type const& id)
    HPX_EXPORT std::string get_locality_name();

    /// \fn future<std::string> get_locality_name(naming::id_type const& id)
    ///
    /// \brief Return the name of the referenced locality.
    ///
    /// This function returns a future referring to the name for the locality
    /// of the given id.
    ///
    /// \param id [in] The global id of the locality for which the name should
    ///           be retrieved
    ///
    /// \returns  This function returns the name for the locality of the given
    ///           id. The name is retrieved from the underlying networking layer
    ///           and may be different for different parcel ports.
    ///
    /// \see      \a std::string get_locality_name()
    HPX_EXPORT future<std::string> get_locality_name(
        naming::id_type const& id);
}

