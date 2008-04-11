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

#ifndef jt_EXTENSIONS_H_
#define jt_EXTENSIONS_H_

#include <vector>
#include <string>
#include <iostream>

struct extensions
{
    typedef std::vector<std::string> array;
    array vals;

    extensions(const std::string & ext = "") {
        if ( !ext.empty())
            vals.push_back(ext);
    }

    extensions & add(const std::string & ext) {
        vals.push_back(ext);
        return *this;
    }
};

inline std::ostream & operator<<(std::ostream & out, const extensions& exts) {
    out << "[";
    for ( extensions::array::const_iterator b = exts.vals.begin(), e = exts.vals.end(); b != e ; ++b) {
        if ( b != exts.vals.begin())
            out << "; ";
        out << "." << (*b);
    }
    out << "]";
    return out;
}

#endif
