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

#ifndef jt_FILE_STATISTICS_h
#define jt_FILE_STATISTICS_h

/** 
    statics (lines of code) from a file
*/
struct file_statistics
{
    file_statistics();

    int commented;
    int empty;
    int code;
    int total;
    
    // this contains the nubmer of chars, once each line is trimmed
    unsigned long non_space_chars;

    void operator+=(const file_statistics& to_add);
};

#endif

