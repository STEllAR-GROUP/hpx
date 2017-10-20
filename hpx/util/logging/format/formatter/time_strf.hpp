// formatter_time.hpp

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


#ifndef JT28092007_formatter_time_strf_HPP_DEFINED
#define JT28092007_formatter_time_strf_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format/formatter/convert_format.hpp>
#include <hpx/util/logging/detail/manipulator.hpp> // is_generic
#include <stdio.h>
#include <time.h>

namespace hpx { namespace util { namespace logging { namespace formatter {


/**
@brief Prefixes the message with the time, by using strftime function.
You pass the format string at construction.

@param msg_type The type that holds your logged message.

@param convert [optional] In case there needs to be a conversion between
std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref hpx::util::logging::optimize::cache_string_one_str
"a cached_string class" (see @ref hpx::util::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::prepend> struct time_strf_t : is_generic {
    typedef convert convert_type;

    /**
        constructs a time_strf object

        @param format the time format , strftime-like
        @param localtime if true, use localtime, otherwise global time
    */
    time_strf_t(const hold_string_type & format, bool localtime)
        : m_format (format), m_localtime (localtime)
    {}

    template<class msg_type> void operator()(msg_type & msg) const {
        char_type buffer[64];
        ::time_t t = ::time (nullptr);
        ::tm t_details = m_localtime ? *localtime( &t) : *gmtime( &t);
        if (0 != strftime (buffer, sizeof (buffer), m_format.c_str (), &t_details))
            convert::write(buffer, msg);
    }

    bool operator==(const time_strf_t & other) const {
        return m_format == other.m_format;
    }

private:
    hold_string_type m_format;
    bool m_localtime;

};



/** @brief time_strf_t with default values. See time_strf_t

@copydoc time_strf_t
*/
typedef time_strf_t<> time_strf;

}}}}

#endif

