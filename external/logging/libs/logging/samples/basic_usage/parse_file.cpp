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

#include "parse_file.h"

// Wherever you use logs, include this ;)
#include "log.h"

#include "util.h"
#include <fstream>

file_statistics parse_file(const fs::path& file) {
    LDBG_ << "Parsing file " << file.string();

    file_statistics stats;
    std::ifstream in( file.string().c_str());
    std::string line;
    bool within_c_comment = false;
    while ( std::getline(in, line) ) {
        trim_str(line);

        if ( line.empty() )
            ++stats.empty;

        bool within_comment = within_c_comment;
        if ( line.size() >= 2)
            if ( line.find("//") == 0)
                within_comment = true;

        if ( line.size() >= 2)
            if ( line.find("/*") == 0) {
                within_comment = true;
                within_c_comment = true;
            }


        if ( within_comment)
            ++stats.commented;
        else {
            if ( !line.empty())
                ++stats.code;
        }

        ++stats.total;
        stats.non_space_chars += (int)line.size();

        // see if C comment has ended
        if ( within_c_comment)
            if ( line.size() >= 2)
                if ( line.substr( line.size() - 2, 2) == "*/")
                    within_c_comment = false;
    }

    if ( stats.total < 1)
        LERR_ << "Could not read from file " << file.string();

    LAPP_ << "File " << file.string() << ":\n"
        << "\n  Code     : " << stats.code
        << "\n  Comments : " << stats.commented
        << "\n  Empty    : " << stats.empty
        << "\n  Total    : " << stats.total
        << "\n"
        << "\n  Avg C/L  : " << (int)((double)stats.non_space_chars / (double)(stats.code + stats.commented))
        ;

    return stats;
}

