// formatter_defaults.hpp

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

#ifndef JT28092007_formatter_defaults_HPP_DEFINED
#define JT28092007_formatter_defaults_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/format.hpp>
#include <hpx/logging/manipulator.hpp>

#include <cstdint>
#include <string>

namespace hpx { namespace util { namespace logging { namespace formatter {

    /**
@brief prefixes each message with an index.

Example:
@code
L_ << "my message";
L_ << "my 2nd message";
@endcode

This will output something similar to:

@code
[1] my message
[2] my 2nd message
@endcode
*/
    struct idx : formatter::manipulator
    {
        idx()
          : value(0ull)
        {
        }

        void operator()(message& str) override
        {
            std::string idx = hpx::util::format("{:016x}", ++value);
            str.prepend_string(idx);
        }

    private:
        std::uint64_t value;
    };

}}}}    // namespace hpx::util::logging::formatter

#endif
