// tss_stream.hpp

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


#ifndef JT28092007_tss_ostringstream_HPP_DEFINED
#define JT28092007_tss_ostringstream_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/tss/tss.hpp>
#include <sstream>

namespace hpx { namespace util { namespace logging {

/**
@brief Represents an ostringstream that takes advantage of TSS (Thread Specific Storage).
       In other words, each thread has its
       own copy of an ostringstream, thus when needed,
       we avoid the cost of re-creating it (it's created only once per thread).
*/
template< class stream = std::basic_ostringstream<char_type> > struct tss_ostringstream {
    typedef stream stream_type;
    typedef hold_string_type string_type;

    tss_ostringstream() {}
    tss_ostringstream(const tss_ostringstream&) {}


    stream_type & get() {
        stream_type & val = *(m_cache.get());
        val.str( HPX_LOG_STR("") );
        return val;
    }

    string_type str() const {
        stream_type & val = *(m_cache.get());
        return val.str();
    }


private:
    mutable tss_value<stream_type> m_cache;
};

template<class stream, class value_type> inline stream& operator
<<( tss_ostringstream<stream> & out, const value_type & val) {
    stream & result = out.get();
    result << val;
    return result;
}

}}}

#endif

