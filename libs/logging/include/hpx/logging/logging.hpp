// logging.hpp

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

#ifndef JT28092007_logging_HPP_DEFINED
#define JT28092007_logging_HPP_DEFINED

#include <hpx/logging/detail/fwd.hpp>
#include <hpx/logging/detail/logger.hpp>
#include <hpx/logging/detail/macros.hpp>
#include <hpx/logging/level.hpp>

namespace hpx { namespace util { namespace logging {

    /**
@file hpx/logging/logging.hpp

Include this file when you're using the logging lib, but don't necessarily want to
use @ref manipulator "formatters and destinations".
If you want to use @ref manipulator "formatters and destinations",
then you can include this one instead:

@code
#include <hpx/logging/format.hpp>
@endcode

*/

}}}    // namespace hpx::util::logging

#endif
