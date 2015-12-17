// format_msg_type.hpp

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


#ifndef JT28092007_format_msg_type_HPP_DEFINED
#define JT28092007_format_msg_type_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/find_gather.hpp>
#ifndef HPX_HAVE_LOG_NO_TSS
  #include <hpx/util/logging/detail/tss/tss_ostringstream.hpp>
#else
  #include <sstream>
#endif

namespace hpx { namespace util { namespace logging {

template<class gather_msg , class write_msg > struct logger ;

namespace formatter {
    /**
    @brief what is the default type of your string,
         in formatter_base ? See HPX_LOG_FORMAT_MSG
    */
    template<class T = override> struct msg_type {
        typedef hold_string_type type;
    };
}

namespace destination {
    /**
    @brief what is the default type of your string,
    in destination_base ? See HPX_LOG_DESTINATION_MSG
    */
    template<class T = override> struct msg_type {
        // by default  - the default string
        typedef hold_string_type type;
    };
}

namespace gather {
    template<class T = override> struct find {
        template<class msg_type> struct from_msg_type {
            typedef typename ::hpx::util::logging::detail
                ::find_gather< std::basic_ostringstream<char_type>,
                  msg_type >::type type;
        };
    };
}

}}}

#endif

