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

#include <hpx/format.hpp>
#include <hpx/logging/detail/fwd.hpp>
#include <hpx/logging/detail/manipulator.hpp>

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


@param convert [optional] In case there needs to be a conversion between
std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref hpx::util::logging::optimize::cache_string_one_str
"a cached_string class" (see @ref hpx::util::logging::optimize "optimize namespace").
*/
    struct idx
      : is_generic
      , formatter::non_const_context<std::uint64_t>
    {
        typedef formatter::non_const_context<std::uint64_t>
            non_const_context_base;

        idx()
          : non_const_context_base(0ull)
        {
        }
        void operator()(msg_type& str) const
        {
            std::string idx = hpx::util::format("{:016x}", ++context());
            str.prepend_string(idx);
        }

        bool operator==(const idx&) const
        {
            return true;
        }
    };

}}}}    // namespace hpx::util::logging::formatter

#endif
