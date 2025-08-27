//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2013 Adrian Serio
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    // Returns the major HPX version.
    HPX_NODISCARD_CORE std::uint8_t major_version();

    // Returns the minor HPX version.
    HPX_NODISCARD_CORE std::uint8_t minor_version();

    // Returns the sub-minor/patch-level HPX version.
    HPX_NODISCARD_CORE std::uint8_t subminor_version();

    // Returns the full HPX version.
    HPX_NODISCARD_CORE std::uint32_t full_version();

    // Returns the full HPX version.
    HPX_NODISCARD_CORE std::string full_version_as_string();

    // Returns the AGAS subsystem version.
    HPX_NODISCARD_CORE std::uint8_t agas_version();

    // Returns the tag.
    HPX_NODISCARD_CORE std::string tag();

    // Returns the HPX full build information string.
    HPX_NODISCARD_CORE std::string full_build_string();

    // Returns the HPX version string.
    HPX_NODISCARD_CORE std::string build_string();

    // Returns the Boost version string.
    HPX_NODISCARD_CORE std::string boost_version();

    // Returns the Boost platform string.
    HPX_NODISCARD_CORE std::string boost_platform();

    // Returns the Boost compiler string.
    HPX_NODISCARD_CORE std::string boost_compiler();

    // Returns the Boost standard library string.
    HPX_NODISCARD_CORE std::string boost_stdlib();

    // Returns the copyright string.
    HPX_NODISCARD_CORE std::string copyright();

    // Returns the full version string.
    HPX_NODISCARD_CORE std::string complete_version();

    // Returns the HPX build type ('Debug', 'Release', etc.)
    HPX_NODISCARD_CORE std::string build_type();

    // Returns the HPX build date and time
    HPX_NODISCARD_CORE std::string build_date_time();

    // Return the HPX configuration information
    HPX_NODISCARD_CORE std::string configuration_string();
}    // namespace hpx
