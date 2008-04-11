// format_fwd_detail.hpp

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


#ifndef JT28092007_format_fwd_detail_HPP_DEFINED
#define JT28092007_format_fwd_detail_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/logging.hpp>
#include <boost/logging/format/optimize.hpp>
#include <boost/logging/gather/ostream_like.hpp>

namespace boost { namespace logging {

namespace writer {
    template<class msg_type, class base_type> struct on_dedicated_thread ;
    template<class base_type> struct ts_write ;

    /** 
        @brief specify thread-safety of your logger_format_write class
    */
    namespace threading {
        /** @brief not thread-safe */
        struct no_ts {};
        /** @brief thread-safe write. All writes are protected by a lock */
        struct ts_write {};
        /** @brief thread-safe write on a dedicated thread. Very efficient. Formatting & writing to destinations happens on the dedicated thread */
        struct on_dedicated_thread {};
    }
}

/** 
@file boost/logging/format_fwd.hpp

Include this file when you're using @ref manipulator "formatters and destinations",
and you want to declare the logger classes, in a header file
(using BOOST_DECLARE_LOG)

Example:

@code
#ifndef LOG_H_header
#define LOG_H_header

#include <boost/logging/logging.hpp>
#include <boost/logging/format/optimize.hpp>

BOOST_LOG_FORMAT_MSG( boost::logging::optimize::cache_string_one_str<> ) 

#if defined(BOOST_LOG_DEFINE_LOGS)
#include <boost/logging/format.hpp>

typedef logger_format_write< > logger_type;
#endif

BOOST_DECLARE_LOG(g_l, logger_type)
BOOST_DECLARE_LOG_FILTER(g_l_filter, level::holder)

#define LDBG_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), debug ) << "[dbg] "
#define LERR_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), error ) << "[ERR] "
#define LAPP_ BOOST_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), info ) << "[app] "

void init_logs();

#endif
@endcode
*/
template<
            class format_base_type = default_, 
            class destination_base_type = default_ ,
            class thread_safety = default_ ,
            class gather = default_,
            class lock_resource = default_
    > struct logger_format_write;


/** 
    dumps the default levels

    Has a static function : dump, which dumps the level as string (works only for the default levels; for any other level, returns "")
*/
struct dump_default_levels {
    static const char_type * dump(::boost::logging::level::type lvl) {
        using namespace ::boost::logging::level;
        switch ( lvl) {
            case debug:     return BOOST_LOG_STR("[debug] ");
            case info:      return BOOST_LOG_STR("[info]  ");
            case warning:   return BOOST_LOG_STR("[warn]  ");
            case error:     return BOOST_LOG_STR("[ERROR] ");
            case fatal:     return BOOST_LOG_STR("[FATAL] ");
            default:        return BOOST_LOG_STR("");
        }
    }
};

/** 
    Specifies the class that will dump the levels. Used by formatter::tag::level class.
*/
template<class T = override> struct dump_level {
    typedef dump_default_levels type;
};


namespace detail {
    // finds the gather type, when using formatting (for logger_format_write)
    template<class gather> struct format_find_gather {
        typedef typename detail::to_override<gather>::type override_;

        // FIXME in the future, I might provide gather as a specific class!
        typedef typename formatter::msg_type<override_>::type msg_type;
        typedef typename ::boost::logging::gather::find<override_>::template from_msg_type<msg_type>::type type;
    };
}


// specialize for logger_format_write
template<class format_base, class destination_base, class thread_safety, class gather, class lock_resource> 
        struct logger_to_gather< logger_format_write<format_base, destination_base, thread_safety, gather, lock_resource> > {

    typedef typename detail::format_find_gather<gather>::type gather_type;
};

namespace writer {
    template<class format_write_ = default_ > struct named_write ;
}

/** @brief named_logger<...>::type finds a logger that uses @ref writer::named_write<> "Named Formatters and Destinations"

@code
#include <boost/logging/format/named_write.hpp>
@endcode

Example:
@code
typedef boost::logging::named_logger<>::type logger_type;
@endcode

Setting the formatters and destinations to write to is extremely simple:

@code
// first param - the formatter(s) , second param : the destination(s)
g_l()->writer().write("%time%($hh:$mm.$ss.$mili) [%idx%] |\n", "cout file(out.txt) debug");
@endcode

To see the syntax, see writer::named_write

*/
template<class gather = default_> struct named_logger {
    typedef typename detail::format_find_gather<gather>::type gather_type;

    /** @copydoc named_logger */
    typedef logger< gather_type, writer::named_write<> > type;
};

}}


#include <boost/logging/detail/scenario.hpp>
#include <boost/logging/detail/tags.hpp>

#endif

