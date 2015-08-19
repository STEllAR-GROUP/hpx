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


#ifndef JT28092007_formatter_time_HPP_DEFINED
#define JT28092007_formatter_time_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format/formatter/convert_format.hpp>
#include <hpx/util/logging/detail/manipulator.hpp> // is_generic
#include <hpx/util/logging/detail/time_format_holder.hpp>

#include <time.h>

namespace hpx { namespace util { namespace logging { namespace formatter {

/**
@brief Prefixes the message with the time. You pass the format string at construction.

It's friendlier than write_time_strf (which uses strftime).

The format can contain escape sequences:
$dd - day, 2 digits
$MM - month, 2 digits
$yy - year, 2 digits
$yyyy - year, 4 digits
$hh - hour, 2 digits
$mm - minute, 2 digits
$ss - second, 2 digits

Example: time("Today is $dd/$MM/$yyyy");

Note: for a high precision clock, try high_precision_time (uses hpx::util::date_time)

@param convert [optional] In case there needs to be a conversion between
std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref hpx::util::logging::optimize::cache_string_one_str
"a cached_string class" (see @ref hpx::util::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::prepend> struct time_t
    : is_generic, non_const_context<hpx::util::logging::detail::time_format_holder> {
    typedef convert convert_type;
    typedef non_const_context<hpx::util::logging::detail::time_format_holder>
        non_const_context_base;

    /**
        constructs a time object
    */
    time_t(const hold_string_type & format) : non_const_context_base(format) {}

    template<class msg_type> void write_time(msg_type & msg, ::time_t val) const {
        char_type buffer[64];

        tm details = *localtime( &val);
        non_const_context_base::context().write_time( buffer,
            details.tm_mday, details.tm_mon + 1, details.tm_year + 1900,
            details.tm_hour, details.tm_min, details.tm_sec);

        convert::write(buffer, msg);
    }

    template<class msg_type> void operator()(msg_type & msg) const {
        ::time_t val = ::time(0);
        write_time(msg, val);
    }

    bool operator==(const time_t & other) const {
        return non_const_context_base::context() ==
            other.non_const_context_base::context();
    }

    /** @brief configure through script

        the string = the time format
    */
    void configure(const hold_string_type & str) {
        non_const_context_base::context().set_format(str);
    }

};



/** @brief time_t with default values. See time_t

@copydoc time_t
*/
typedef time_t<> time;

}}}}

#endif

