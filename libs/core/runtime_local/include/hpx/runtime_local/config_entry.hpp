//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    /// Retrieve the string value of a configuration entry given by \p key.
    HPX_CORE_EXPORT std::string get_config_entry(
        std::string const& key, std::string const& dflt);
    /// Retrieve the integer value of a configuration entry given by \p key.
    HPX_CORE_EXPORT std::string get_config_entry(
        std::string const& key, std::size_t dflt);

    /// Set the string value of a configuration entry given by \p key.
    HPX_CORE_EXPORT void set_config_entry(
        std::string const& key, std::string const& value);
    /// Set the integer value of a configuration entry given by \p key.
    HPX_CORE_EXPORT void set_config_entry(
        std::string const& key, std::size_t value);

    /// Set the string value of a configuration entry given by \p key.
    HPX_CORE_EXPORT void set_config_entry_callback(std::string const& key,
        hpx::function<void(std::string const&, std::string const&)> const&
            callback);
}    // namespace hpx
