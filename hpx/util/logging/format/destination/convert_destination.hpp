// convert_destination.hpp

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


#ifndef JT28092007_convert_destination_HPP_DEFINED
#define JT28092007_convert_destination_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>
#include <ostream>
#include <string>

namespace hpx { namespace util { namespace logging { namespace destination {

template<class t> struct into {};

/**
@brief Allows writing messages to destinations

It has 2 function overloads:
- write(message, output) - writes the given message, to the given output
- do_convert(message, into<other_type>() );

FIXME
*/
namespace convert {
    template<class obj> inline void write(const obj & m,
        std::ostream & out) {
        out << m;
    }

    inline void write(const char* m,
        std::ostream & out) {
        out << m;
    }

    inline const char * do_convert(const char * c,
        const into<const char*> &) { return c; }
    inline const char * do_convert(const std::string & s,
        const into<const char* > &) { return s.c_str(); }

    inline const std::string &
        do_convert(const std::string & s,
            const into< std::string > &) {
        return s;
    }
}

}}}}

#endif
