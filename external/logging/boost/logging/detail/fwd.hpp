// fwd.hpp

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


#ifndef JT28092007_fwd_HPP_DEFINED
#define JT28092007_fwd_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <boost/logging/detail/util.hpp>
#include <boost/logging/detail/macros.hpp>

#include <boost/logging/detail/ts/ts.hpp>
#include <boost/logging/detail/ts/ts_resource.hpp>

#include <boost/logging/defaults.hpp>

// minimize inclusion of STL headers in our headers!!!
#include <string>

/* The following BOOST_LOG_STR("this " "and that") still doesn't work.
#define BOOST_LOG_HOLDER2(x) x, L ## x
#define BOOST_LOG_HOLDER(x) BOOST_LOG_HOLDER2(x)
#define BOOST_LOG_STR(x)      (const ::boost::logging::char_type*)ansi_unicode_char_holder ( BOOST_LOG_HOLDER(x) )
*/
#define BOOST_LOG_STR(x)      (const ::boost::logging::char_type*)::boost::logging::ansi_unicode_char_holder ( x, L ## x )


/* 
    Important: we define here only the things that are needed by ALL OF THE LIBRARY.
    So be very careful when modifying this file - we don't want any circular dependencies!

    If unsure where to place something, place it logging.hpp!
*/

namespace boost { namespace logging {
    // see our types
    typedef types<override>::char_type char_type;
    typedef types<override>::hold_string_type hold_string_type;
    typedef types<override>::filter_type filter_type;
    typedef types<override>::mutex mutex;
    typedef types<override>::level_holder_type level_holder_type;

    namespace writer {}



    /* 
        just in case you're doing a typo - "write" instead of "writer"
    */
    namespace write = writer;



/** 
@page dealing_with_flags Dealing with flags.

Some classes have extra settings. You can specify these settings in the class'es constructor.
When setting a certain value, there's a very simple pattern:

@code
some_object obj(..., some_object_settings().setting1(value1).setting2(value2)....);
@endcode

Example:

@code
using namespace destination;
file f("out.txt", file_settings.initial_overwrite(true).do_append(false) );
@endcode

*/


namespace detail {
    template<class self_type, class type> struct flag_with_self_type {
        flag_with_self_type(self_type * self, const type& val = type() ) : m_val(val), m_self(self) {}
        flag_with_self_type(const flag_with_self_type & other) : m_val(other.m_val) {}

        const type & operator()() const { return m_val; }
        self_type & operator()(const type & val) {
            m_val = val; return *m_self;
        }

        void operator=(const self_type & other) {
            m_val = other.m_val;
        }

    private:
        type m_val;
        self_type * m_self;
    };

    /** 
        @brief Can hold a flag. See dealing_with_flags
    */
    template<class self_type> struct flag {
        template<class val_type> struct t : flag_with_self_type<self_type,val_type> {
            typedef flag_with_self_type<self_type,val_type> flag_base_type;
            t(self_type * self, const val_type& val = val_type() ) : flag_base_type(self,val) {}
        };
    };
}

}}


#endif

