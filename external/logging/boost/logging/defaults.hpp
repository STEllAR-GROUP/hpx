// defaults.hpp

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

#ifndef JT28092007_defaults_HPP_DEFINED
#define JT28092007_defaults_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/detail/ts/ts.hpp>

#include <string>
#include <boost/config.hpp>

namespace boost { namespace logging {

/** 
@page override_defaults Defaults, and overriding them.

The logging lib has a few default types, used throughout the lib. They are:
- @c char_type - the char type used throught the lib; by default, it's @c char
- @c hold_string_type - the type used to hold a string; by default, it's @c std::string
- @c filter_type - the default filter; by default, it's filter::no_ts
- @c lock_resource - used to lock resources for access. See locker namespace.
- @c mutex - the mutex class used throughout the library. By default, it's mutex_win32 for Windows, or mutex_posix for POSIX 

They are all present in @c default_types structure.

If you want to override any of the above, you should do the following:
- before including anything from Boost Logging Library, <tt>\#include <boost/logging/defaults.hpp> </tt>
- override the types
- do <tt>\#include <boost/logging/logging.hpp> </tt>
      
Example:

@code
    #include <boost/logging/defaults.hpp>

    namespace boost { namespace logging {
        struct types<override> : default_types {
            // define your types
            typedef wchar_t char_type;
            // etc.
        };
    }}

    #include <boost/logging/logging.hpp>
@endcode


*/

// define BOOST_LOG_USE_WCHAR_T if you want your char type to be 'wchar_t'

#if defined(BOOST_WINDOWS) && !defined(BOOST_LOG_DONOT_USE_WCHAR_T)
#if defined( UNICODE) || defined(_UNICODE)
#undef BOOST_LOG_USE_WCHAR_T
#define BOOST_LOG_USE_WCHAR_T
#endif
#endif




    // forward defines

    namespace filter {
        struct no_ts;
    }

    namespace level {
        struct holder_no_ts ;
        struct holder_ts ;
        template<int> struct holder_tss_with_cache ;
    }

    namespace locker {
        template<class type, class mutex > struct ts_resource ;
        template<class , int, class> struct tss_resource_with_cache ;
    }


    struct default_types {
#ifdef BOOST_LOG_USE_WCHAR_T
        typedef wchar_t char_type;
#else
        typedef char char_type;
#endif
        // this is the type we use to hold a string, internally
        typedef std::basic_string<char_type> hold_string_type;

        // default filter type
        typedef filter::no_ts filter_type;

        typedef level::holder_no_ts level_holder_type;

        struct lock_resource {
            template<class lock_type> struct finder {
//#if !defined( BOOST_LOG_NO_TSS) && defined(BOOST_WINDOWS)
                // on Windows, I've tested the threading
//                typedef typename locker::tss_resource_with_cache<lock_type, 5, boost::logging::threading::mutex > type;
//#else
                typedef typename locker::ts_resource<lock_type, boost::logging::threading::mutex > type;
//#endif
            };
        };

        typedef boost::logging::threading::mutex mutex;
    };

    // FIXME we need defaults for different scenarios!
    template<class T> struct types : default_types {
    };
}}

#endif

