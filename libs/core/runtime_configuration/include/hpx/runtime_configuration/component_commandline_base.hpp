//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file component_commandline_base.hpp
/// \page hpx::components::component_commandline_base
/// \headerfile hpx/components.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/program_options.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    ////////////////////////////////////////////////////////////////////////////
    /// The \a component_commandline_base has to be used as a base class for all
    /// component command-line line handling registries.
    HPX_CXX_CORE_EXPORT struct HPX_CORE_EXPORT component_commandline_base
    {
        virtual ~component_commandline_base() = default;

        /// \brief Return any additional command line options valid for this
        ///        component
        ///
        /// \return The module is expected to fill a options_description object
        ///         with any additional command line options this component will
        ///         handle.
        ///
        /// \note   This function will be executed by the runtime system
        ///         during system startup.
        virtual hpx::program_options::options_description
        add_commandline_options() = 0;
    };
}    // namespace hpx::components
