// logging.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_logging_HPP_DEFINED
#define JT28092007_logging_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/detail/fwd.hpp>
#include <boost/logging/detail/filter.hpp>
#include <boost/logging/detail/logger.hpp>
#include <boost/logging/detail/log_keeper.hpp>
#include <boost/logging/detail/macros.hpp>
#include <boost/logging/detail/tss/tss.hpp>
#include <boost/logging/detail/level.hpp>
#include <boost/logging/detail/scoped_log.hpp>

// just in case we might think of using formatters
#include <boost/logging/detail/format_msg_type.hpp>

namespace boost { namespace logging {

/** 
@file boost/logging/logging.hpp

Include this file when you're using the logging lib, but don't necessarily want to use @ref manipulator "formatters and destinations".
If you want to use @ref manipulator "formatters and destinations", then you can include this one instead:

@code
#include <boost/logging/format_fwd.hpp>
@endcode

*/

}}

#endif

