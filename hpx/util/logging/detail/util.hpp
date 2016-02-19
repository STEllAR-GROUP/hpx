// detail/util.hpp

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


#ifndef JT28092007_detail_util_HPP_DEFINED
#define JT28092007_detail_util_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif



/*
    make sure we don't need any of our headers included from here!
    we're included from fwd.hpp!
*/

namespace hpx { namespace util { namespace logging {

    struct override {};

    struct default_ {};
    template<class param, class default_type> struct use_default
    { typedef param type; };
    template<class default_type> struct use_default<default_, default_type>
    { typedef default_type type; };

    struct void_ {};


    namespace detail {
        /** this is just a simple way to always return override; however,
        in this case we postpone the instantiation
         until our template parameter is known


        For instance:
        @code
        typedef typename formatter::msg_type<override>::type msg_type;
        @endcode

        would compute msg_type right now; however, we want the compiler to wait,
        until the user has actually set the msg_type,
        for example, using the HPX_LOG_FORMAT_MSG macro. Thus, we do:

        @code
        typedef typename detail::to_override<format_base>::type T;
        typedef typename formatter::msg_type<T>::type msg_type;
        @endcode
        */
        template<class> struct to_override { typedef override type; };
        template<> struct to_override<void_> { typedef void_ type; };
    }


    struct ansi_unicode_char_holder {
        const char * str;
        const wchar_t * wstr;
        ansi_unicode_char_holder(const char * str_, const wchar_t * wstr_)
            : str(str_), wstr(wstr_) {}

        operator const char*() const { return str; }
        operator const wchar_t*() const { return wstr; }
    };

}}}


#define HPX_LOG_CONCATENATE2(a,b) a ## b

#define HPX_LOG_CONCATENATE(a,b) HPX_LOG_CONCATENATE2(a,b)


#endif

