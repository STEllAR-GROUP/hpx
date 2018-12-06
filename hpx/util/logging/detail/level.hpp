// level.hpp

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


#ifndef JT28092007_level_HPP_DEFINED
#define JT28092007_level_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>

namespace hpx { namespace util { namespace logging {

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
namespace level {
    /** the higher the level, the more critical the error */
    typedef unsigned int type;

    enum {
        disable_all = -1,
        enable_all = 0,
        debug = 1000,
        info = 2000,
        warning = 3000,
        error = 4000,
        fatal = 5000,
        always = 6000
    };

    /**
        @brief Filter - holds the level.

        Holds the level, and tells you if a specific level is enabled.
        It does this in a non-thread-safe way.

        If you change set_enabled() while program is running,
        it can take a bit to propagate
        between threads. Most of the time, this should be acceptable.
    */
    struct holder {
        holder(type default_level = enable_all) : m_level(default_level) {}
        bool is_enabled(type level) const { return level >= m_level; }
        void set_enabled(type level) {
            m_level = level;
        }
    private:
        type m_level;
    };

    /**
        @brief Filter - holds the level
        - and tells you at compile time if a filter is enabled or not.

        Fix (compile time) holder
    */
    template<int fix_level = debug> struct holder_compile_time {
        static bool is_enabled(type level) {
            return fix_level >= level;
        }
    };
} // namespace level

}}}

#endif
