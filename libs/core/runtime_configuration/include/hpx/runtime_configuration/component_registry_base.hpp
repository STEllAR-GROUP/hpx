//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_configuration/static_factory_data.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_registry_base has to be used as a base class for all
    /// component registries.
    HPX_CXX_EXPORT struct HPX_CORE_EXPORT component_registry_base
    {
        virtual ~component_registry_base() = default;

        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in, out] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all components implemented in this
        ///                 module.
        /// \param filepath  [in]
        /// \param is_static [in]
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        virtual bool get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath, bool is_static = false) = 0;

        /// \brief Register the component type represented by this component
        virtual void register_component_type() = 0;
    };
}    // namespace hpx::components
