/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

#ifndef jt_UTIL_h
#define jt_UTIL_h

#include <string>

std::string lo_case( const std::string & str) ;
void trim_str(std::string & str) ;
void str_replace( std::string & str, const std::string & find, const std::string & replace) ;

#endif
