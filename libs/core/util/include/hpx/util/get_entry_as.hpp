/*=============================================================================
    Copyright (c) 2014 Anton Bikineev

//  SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#pragma once

#include <hpx/util/from_string.hpp>

#include <string>
#include <type_traits>

namespace hpx::util {

    template <typename DestType, typename Config,
        std::enable_if_t<!std::is_same_v<DestType, std::string>, bool> = false>
    DestType get_entry_as(
        Config const& config, std::string const& key, DestType const& dflt)
    {
        std::string const& entry = config.get_entry(key, "");
        if (entry.empty())
            return dflt;
        return from_string<DestType>(entry, dflt);
    }

    template <typename DestType, typename Config,
        std::enable_if_t<std::is_same_v<DestType, std::string>, bool> = false>
    DestType get_entry_as(
        Config const& config, std::string const& key, DestType const& dflt)
    {
        return config.get_entry(key, dflt);
    }
}    // namespace hpx::util
