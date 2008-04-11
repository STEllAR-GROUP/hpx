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

#include "dir_spec.h"

// Wherever you use logs, include this ;)
#include "log.h"

#include "parse_file.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

dir_spec::dir_spec(const std::string & path, const extensions & ext) : m_path(path), m_ext(ext), m_file_count(0) {
    LAPP_ << "Parsing dir : " << fs::system_complete(path).string() << ", looking for " << ext;
}


dir_spec::~dir_spec(void)
{
}

void dir_spec::iterate() {
    iterate_dir( m_path);

    LAPP_ << "\n\nAll Files :\n"
        << "\n  No of Files : " << m_file_count
        << "\n"
        << "\n  Code        : " << m_stats.code
        << "\n  Comments    : " << m_stats.commented
        << "\n  Empty       : " << m_stats.empty
        << "\n  Total       : " << m_stats.total
        << "\n"
        << "\n  Avg C/L     : " << (int)((double)m_stats.non_space_chars / (double)(m_stats.code + m_stats.commented))
        ;

}

void dir_spec::iterate_dir(const fs::path & dir) {
    LDBG_ << "Parsing dir " << fs::system_complete(dir).string();

    if ( !fs::exists( dir) ) {
        LERR_ << "Dir " << dir.string() << " does not exist anymore!";
        return ;
    }

    for ( fs::directory_iterator b(dir), e; b != e; ++b) {
        if ( fs::is_directory(*b))
            iterate_dir(*b);
        else {
            // file
            bool matches_ext = false;
            for ( extensions::array::const_iterator b_ext = m_ext.vals.begin(), e_ext = m_ext.vals.end(); b_ext != e_ext; ++b_ext)
                if ( fs::extension(*b) == "." + *b_ext) {
                    matches_ext = true;
                    break;
                }

            if ( matches_ext) {
                m_stats += parse_file(*b);
                ++m_file_count;
            }
            else
                LDBG_ << "Ignoring file " << b->string();
        }
    }

}

