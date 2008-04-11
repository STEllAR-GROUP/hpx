// formatter_defaults.hpp

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


#ifndef JT28092007_formatter_defaults_HPP_DEFINED
#define JT28092007_formatter_defaults_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/logging/detail/fwd.hpp>
#include <boost/logging/detail/manipulator.hpp>
#include <boost/logging/format/formatter/convert_format.hpp>
#include <boost/logging/format/formatter/time.hpp>
#include <boost/logging/format/formatter/time_strf.hpp>
#include <boost/logging/format/formatter/spacer.hpp>
#include <boost/logging/format/formatter/thread_id.hpp>
#include <stdio.h>
#include <time.h>
#include <sstream>

namespace boost { namespace logging { namespace formatter {


/** 
@brief prefixes each message with an index. 

Example:
@code
L_ << "my message";
L_ << "my 2nd message";
@endcode

This will output something similar to:

@code
[1] my message
[2] my 2nd message
@endcode


@param convert [optional] In case there needs to be a conversion between std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref boost::logging::optimize::cache_string_one_str "a cached_string class" (see @ref boost::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::prepend> struct idx_t : is_generic, formatter::non_const_context<int>, boost::logging::op_equal::always_equal  {
    typedef formatter::non_const_context<int> non_const_context_base;
    typedef convert convert_type;

    idx_t() : non_const_context_base((int)0) {}
    template<class msg_type> void operator()(msg_type & str) const {
        std::basic_ostringstream<char_type> idx;
        idx << ++context() ;

        convert::write( idx.str(), str );
    }
};


/** 
@brief Appends a new line

@param convert [optional] In case there needs to be a conversion between std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref boost::logging::optimize::cache_string_one_str "a cached_string class" (see @ref boost::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::append> struct append_newline_t : is_generic, boost::logging::op_equal::always_equal {
    typedef convert convert_type;
    template<class msg_type> void operator()(msg_type & str) const {
        convert::write( BOOST_LOG_STR("\n"), str );
    }
};


/** 
@brief Appends a new line, if not already there

@param convert [optional] In case there needs to be a conversion between std::(w)string and the string that holds your logged message. See convert_format.
For instance, you might use @ref boost::logging::optimize::cache_string_one_str "a cached_string class" (see @ref boost::logging::optimize "optimize namespace").
*/
template<class convert = do_convert_format::append> struct append_newline_if_needed_t : is_generic, boost::logging::op_equal::always_equal {
    typedef convert convert_type;

    template<class msg_type> void operator()(msg_type & str) const {
        bool is_needed = true;
        if ( ! convert::get_underlying_string(str).empty())
            if ( *(convert::get_underlying_string(str).rbegin()) == '\n')
                is_needed = false;

        if ( is_needed)
            convert::write( BOOST_LOG_STR("\n"), str );
    }
};



/** @brief idx_t with default values. See idx_t

@copydoc idx_t
*/
typedef idx_t<> idx;

/** @brief append_newline_t with default values. See append_newline_t

@copydoc append_newline_t
*/
typedef append_newline_t<> append_newline;

/** @brief append_newline_if_needed_t with default values. See append_newline_if_needed_t

@copydoc append_newline_if_needed_t
*/
typedef append_newline_if_needed_t<> append_newline_if_needed;

}}}

#endif

