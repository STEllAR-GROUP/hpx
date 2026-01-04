////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/preprocessor.hpp>

#include <string>

namespace hpx::util {

    // set and query the prefix as configured at compile time
    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT void set_hpx_prefix(
        char const* prefix) noexcept;
    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT char const*
    hpx_prefix() noexcept;

    // return the installation path of the specified module
    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT std::string find_prefix(
        std::string const& library = "hpx");

    // return a list of paths delimited by HPX_INI_PATH_DELIMITER
    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT std::string find_prefixes(
        std::string const& suffix, std::string const& library = "hpx");

    // return the full path of the current executable
    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT std::string
    get_executable_filename(char const* argv0 = nullptr);
    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT std::string
    get_executable_prefix(char const* argv0 = nullptr);
}    // namespace hpx::util
