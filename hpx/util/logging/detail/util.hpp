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

/*
    make sure we don't need any of our headers included from here!
    we're included from fwd.hpp!
*/

namespace hpx { namespace util { namespace logging {

    struct default_ {};
    template<class param, class default_type> struct use_default
    { typedef param type; };
    template<class default_type> struct use_default<default_, default_type>
    { typedef default_type type; };

    struct void_ {};

}}}

#endif
