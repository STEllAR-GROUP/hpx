// high_precision_time.hpp

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


#ifndef JT28092007_high_precision_time_HPP_DEFINED
#define JT28092007_high_precision_time_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>

#include <hpx/util/logging/format/formatter/convert_format.hpp>
#include <hpx/util/logging/detail/manipulator.hpp> // is_generic
#include <hpx/util/logging/detail/time_format_holder.hpp>

#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>


namespace hpx { namespace util { namespace logging { namespace formatter {


/**
@brief Prefixes the message with a high-precision time (.
You pass the format string at construction.

@code
#include <hpx/util/logging/format/formatter/high_precision_time.hpp>
@endcode

Internally, it uses hpx::util::date_time::microsec_time_clock.
So, our precision matches this class.

The format can contain escape sequences:
$dd - day, 2 digits
$MM - month, 2 digits
$yy - year, 2 digits
$yyyy - year, 4 digits
$hh - hour, 2 digits
$mm - minute, 2 digits
$ss - second, 2 digits
$mili - milliseconds
$micro - microseconds (if the high precision clock allows; otherwise, it pads zeros)
$nano - nanoseconds (if the high precision clock allows; otherwise, it pads zeros)


Example:

@code
high_precision_time("$mm:$ss:$micro");
@endcode

@param convert [optional] In case there needs to be a conversion between
std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref hpx::util::logging::optimize::cache_string_one_str
"a cached_string class" (see @ref hpx::util::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::prepend> struct high_precision_time_t
    : is_generic, non_const_context<hpx::util::logging::detail::time_format_holder> {
    typedef convert convert_type;
    typedef non_const_context<hpx::util::logging::detail::time_format_holder>
        non_const_context_base;

    /**
        constructs a high_precision_time object
    */
    high_precision_time_t(const hold_string_type & format)
        : non_const_context_base(format) {}

    template<class msg_type>
    void write_high_precision_time(msg_type & msg, ::boost::posix_time::ptime val)
        const {
        char_type buffer[64];

        int nanosecs = static_cast<int>(val.time_of_day().fractional_seconds()); //-V807
        int digits = static_cast<int>(val.time_of_day().num_fractional_digits());
        switch ( digits) {
        case 0: break; // no high precision at all
        case 1: nanosecs *= 100000000; break;
        case 2: nanosecs *= 10000000; break;
        case 3: nanosecs *= 1000000; break;
        case 4: nanosecs *= 100000; break;
        case 5: nanosecs *= 10000; break;
        case 6: nanosecs *= 1000; break;
        case 7: nanosecs *= 100; break;
        case 8: nanosecs *= 10; break;
        case 9: break;
        default:
            while ( digits > 9) {
                nanosecs /= 10; digits--;
            }
            break;
        }

        non_const_context_base::context().write_time( buffer,
            val.date().day(), //-V807
            val.date().month(),
            val.date().year(),
            val.time_of_day().hours(),
            val.time_of_day().minutes(),
            val.time_of_day().seconds(),
            nanosecs / 1000000,
            nanosecs / 1000,
            nanosecs
            );

        convert::write(buffer, msg);
    }

    template<class msg_type> void operator()(msg_type & msg) const {
        write_high_precision_time(msg,
            ::boost::posix_time::microsec_clock::local_time() );
    }

    bool operator==(const high_precision_time_t & other) const {
        return non_const_context_base::context() ==
            other.non_const_context_base::context() ;
    }

    /** @brief configure through script

        the string = the time format
    */
    void configure(const hold_string_type & str) {
        non_const_context_base::context().set_format(str);
    }


};



/** @brief high_precision_time_t with default values. See high_precision_time_t

@copydoc high_precision_time_t
*/
typedef high_precision_time_t<> high_precision_time;

}}}}

#endif

