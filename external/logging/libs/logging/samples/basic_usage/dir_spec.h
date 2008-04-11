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

#ifndef jt_DIR_SPEC_H
#define jt_DIR_SPEC_H

#include "extensions.h"
#include <boost/filesystem/path.hpp>
namespace fs = boost::filesystem;

#include "file_statistics.h"

/** 
    Contains a directory specification 
    - what directories we're to search
    - what type of files we're to seach
*/
struct dir_spec
{
    dir_spec(const std::string & path, const extensions & ext);
    ~dir_spec(void);

    void iterate();

private:
    void iterate_dir(const fs::path & dir);

private:
    std::string m_path;
    extensions m_ext;

    file_statistics m_stats;
    int m_file_count;
};

#endif
