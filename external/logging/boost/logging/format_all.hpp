// format_all.hpp

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


#ifndef JT28092007_format_all_HPP_DEFINED
#define JT28092007_format_all_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/detail/fwd.hpp>
#include <boost/logging/format.hpp>
#include <boost/logging/format/formatter/thread_id.hpp>
#include <boost/logging/format/formatter/time.hpp>
#include <boost/logging/format/destination/file.hpp>
#include <boost/logging/format/destination/rolling_file.hpp>


// not tested yet
//#include <boost/logging/format/destination/shared_memory.hpp>


#endif

