//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_configuration/runtime_configuration_fwd.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::plugins {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_registry_base has to be used as a base class for all
    /// plugin registries.
    HPX_CXX_EXPORT struct HPX_CORE_EXPORT plugin_registry_base
    {
        virtual ~plugin_registry_base() = default;

        /// Return the configuration information for any plugin implemented by
        /// this module
        ///
        /// \param fillini  [in, out] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all plugins implemented in this module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        virtual bool get_plugin_info(std::vector<std::string>& fillini) = 0;

        virtual void init(
            int* /*argc*/, char*** /*argv*/, util::runtime_configuration&)
        {
        }
    };
}    // namespace hpx::plugins
