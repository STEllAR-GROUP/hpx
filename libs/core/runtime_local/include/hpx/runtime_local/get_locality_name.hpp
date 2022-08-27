//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime_local/get_locality_name.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/futures.hpp>

#include <string>

namespace hpx {

    namespace detail {

        HPX_CORE_EXPORT std::string get_locality_name();
    }    // namespace detail

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
    /// \see      \a future<std::string> get_locality_name(hpx::id_type const& id)
    HPX_CORE_EXPORT std::string get_locality_name();
}    // namespace hpx
