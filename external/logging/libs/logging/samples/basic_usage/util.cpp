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

#include "util.h"
#include <algorithm>

std::string lo_case( const std::string & str) {
    std::string lo;
    lo.resize( str.length());
    std::transform( str.begin(), str.end(), lo.begin(), tolower);
    return lo;
}

void trim_str(std::string & str) {
    std::string::iterator begin = str.begin(), end = str.end();
    while ( begin < end)
        if ( isspace(*begin)) ++begin;
        else break;

    while ( begin < end)
        if ( isspace(end[-1]) ) --end;
        else break;

    str = std::string(begin, end);
}

void str_replace( std::string & str, const std::string & find, const std::string & replace) {
    size_t pos = 0;
    while ( true) {
        size_t next = str.find( find, pos);
        if ( next == std::string::npos) 
            break;
        str.erase( next, find.length() );
        str.insert( next, replace);
        pos = next + replace.size();
    }
}
