// format_ts.hpp

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

#ifndef JT28092007_format_ts_HPP_DEFINED
#define JT28092007_format_ts_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/format.hpp>

#if !defined(HPX_HAVE_LOG_NO_TS)

#include <hpx/util/logging/writer/ts_write.hpp>
#include <hpx/util/logging/writer/on_dedicated_thread.hpp>

namespace hpx { namespace util { namespace logging {

/**
@file hpx/util/logging/format_ts.hpp

Include this file when you're using @ref manipulator "formatters and destinations",
and you want to define the logger classes, in a source file
(using HPX_DEFINE_LOG) and you've decided to use some form of thread-safety

*/

}}}

#endif

#endif

