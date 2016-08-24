//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2013 Adrian Serio
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_VERSION_AUG_18_2011_0854PM)
#define HPX_VERSION_AUG_18_2011_0854PM

#include <hpx/config.hpp>
#include <hpx/util_fwd.hpp>

#include <cstdint>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // Returns the major HPX version.
    HPX_EXPORT std::uint8_t major_version();

    // Returns the minor HPX version.
    HPX_EXPORT std::uint8_t minor_version();

    // Returns the sub-minor/patch-level HPX version.
    HPX_EXPORT std::uint8_t subminor_version();

    // Returns the full HPX version.
    HPX_EXPORT std::uint32_t full_version();

    // Returns the full HPX version.
    HPX_EXPORT std::string full_version_as_string();

    // Returns the AGAS subsystem version.
    HPX_EXPORT std::uint8_t agas_version();

    // Returns the tag.
    HPX_EXPORT std::string tag();

    // Returns the HPX full build information string.
    HPX_EXPORT std::string full_build_string();

    // Returns the HPX version string.
    HPX_EXPORT std::string build_string();

    // Returns the Boost version string.
    HPX_EXPORT std::string boost_version();

    // Returns the Boost platform string.
    HPX_EXPORT std::string boost_platform();

    // Returns the Boost compiler string.
    HPX_EXPORT std::string boost_compiler();

    // Returns the Boost standard library string.
    HPX_EXPORT std::string boost_stdlib();

    // Returns the copyright string.
    HPX_EXPORT std::string copyright();

    // Returns the full version string.
    HPX_EXPORT std::string complete_version();

    // Returns the HPX build type ('Debug', 'Release', etc.)
    HPX_EXPORT std::string build_type();

    // Returns the HPX build date and time
    HPX_EXPORT std::string build_date_time();

    // Return the HPX configuration information
    HPX_EXPORT std::string configuration_string();

    // Return the HPX runtime configuration information
    HPX_EXPORT std::string runtime_configuration_string(
        util::command_line_handling const& cfg);
}

#endif
