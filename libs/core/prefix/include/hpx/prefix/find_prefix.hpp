////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <string>

namespace hpx::util {

    // set and query the prefix as configured at compile time
    HPX_CORE_EXPORT void set_hpx_prefix(const char* prefix) noexcept;
    [[nodiscard]] HPX_CORE_EXPORT char const* hpx_prefix() noexcept;

    // return the installation path of the specified module
    [[nodiscard]] HPX_CORE_EXPORT std::string find_prefix(
        std::string const& library = "hpx");

    // return a list of paths delimited by HPX_INI_PATH_DELIMITER
    [[nodiscard]] HPX_CORE_EXPORT std::string find_prefixes(
        std::string const& suffix, std::string const& library = "hpx");

    // return the full path of the current executable
    [[nodiscard]] HPX_CORE_EXPORT std::string get_executable_filename(
        char const* argv0 = nullptr);
    [[nodiscard]] HPX_CORE_EXPORT std::string get_executable_prefix(
        char const* argv0 = nullptr);
}    // namespace hpx::util

// The HPX runtime needs to know where to look for the HPX ini files if no ini
// path is specified by the user (default in $HPX_LOCATION/share/hpx-1.0.0/ini).
// Also, the default component path is set within the same prefix.

// clang-format off
#define HPX_BASE_DIR_NAME                                                      \
    "hpx-"                                                                     \
    HPX_PP_STRINGIZE(HPX_VERSION_MAJOR) "."                                    \
    HPX_PP_STRINGIZE(HPX_VERSION_MINOR) "."                                    \
    HPX_PP_STRINGIZE(HPX_VERSION_SUBMINOR) /**/
// clang-format on

#if !defined(HPX_DEFAULT_INI_PATH)
#define HPX_DEFAULT_INI_PATH                                                   \
    hpx::util::find_prefixes("/share/" HPX_BASE_DIR_NAME "/ini") /**/
#endif
#if !defined(HPX_DEFAULT_INI_FILE)
#define HPX_DEFAULT_INI_FILE                                                   \
    hpx::util::find_prefixes("/share/" HPX_BASE_DIR_NAME "/hpx.ini") /**/
#endif
#if !defined(HPX_DEFAULT_COMPONENT_PATH)
#define HPX_DEFAULT_COMPONENT_PATH hpx::util::find_prefixes("/hpx") /**/
#endif
