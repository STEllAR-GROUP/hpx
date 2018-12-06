// cache_before_init_macros.hpp

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


#ifndef JT28092007_cache_before_init_macros_HPP_DEFINED
#define JT28092007_cache_before_init_macros_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {

///////////////////////////////////////////////////////////////////////////////////
// Messages that were logged before initializing the log
// - cache the message (and I'll write it even if the filter is turned off)

#define HPX_LOG_USE_LOG(l, do_func, is_log_enabled) if ( !(is_log_enabled) ) ; \
        else l-> do_func


}}}

#endif
