// format_fwd.hpp

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


// this needs to be fixed!
#ifndef JT28092007_format_fwd_HPP_DEFINED
#define JT28092007_format_fwd_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/detail/format_fwd_detail.hpp>

#if !defined( BOOST_LOG_COMPILE_FAST)
// slow compile
#include <boost/logging/format.hpp>
#endif


#endif

