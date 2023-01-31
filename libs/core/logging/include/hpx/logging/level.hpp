// level.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#pragma once

#include <hpx/config.hpp>

#include <string_view>

namespace hpx::util::logging {

    /**
    @brief Handling levels - classes that can hold and/or deal with levels
    - filters and level holders

    By default we have these levels:

        - debug (smallest level),
        - info,
        - warning ,
        - error ,
        - fatal (highest level)

    Depending on which level is enabled for your application,
    some messages will reach the log: those
    messages having at least that level. For instance, if info level is enabled, all
    logged messages will reach the log.
    If warning level is enabled, all messages are logged, but the warnings.
    If debug level is enabled, messages that have levels debug,
    error, fatal will be logged.

    */
    enum class level : unsigned int
    {
        disable_all = static_cast<unsigned int>(-1),
        enable_all = 0,
        debug = 1000,
        info = 2000,
        warning = 3000,
        error = 4000,
        fatal = 5000,
        always = 6000
    };

    ////////////////////////////////////////////////////////////////////////////
    HPX_CORE_EXPORT void format_value(
        std::ostream& os, std::string_view spec, level value);
}    // namespace hpx::util::logging
