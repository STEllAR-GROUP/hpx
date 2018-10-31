// log_keeper.hpp

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


#ifndef JT28092007_log_keeper_HPP_DEFINED
#define JT28092007_log_keeper_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>

#include <cstdint>

namespace hpx { namespace util { namespace logging {


/**
    @brief Ensures the log is created before main(), even if not used before main

    We need this, so that we won't run into multi-threaded issues while
    the log is created
    (in other words, if the log is created before main(),
    we can safely assume there's only one thread running,
    thus no multi-threaded issues)
*/
struct ensure_early_log_creation {
    template<class type> ensure_early_log_creation ( type & log) {
    typedef std::int64_t long_type ;
        long_type ignore = reinterpret_cast<long_type>(&log);
        // we need to force the compiler to force creation of the log
        if ( time(nullptr) < 0)
            if ( time(nullptr) < (time_t)ignore) {
                printf("LOGGING LIB internal error - should NEVER happen. \
                    Please report this to the author of the lib");
                exit(0);
            }
    }
};


/**
    @brief Ensures the filter is created before main(), even if not used before main

    We need this, so that we won't run into multi-threaded issues while
    the filter is created
    (in other words, if the filter is created before main(),
    we can safely assume there's only one thread running,
    thus no multi-threaded issues)
*/
typedef ensure_early_log_creation ensure_early_filter_creation;


}}}

#endif
