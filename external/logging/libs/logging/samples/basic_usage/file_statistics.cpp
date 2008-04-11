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

#include "file_statistics.h"

file_statistics::file_statistics() : commented(0), empty(0), code(0), total(0), non_space_chars(0) {
}

void file_statistics::operator+=(const file_statistics& to_add) {
    commented += to_add.commented;
    empty += to_add.empty;
    code += to_add.code;
    total += to_add.total;
    non_space_chars += to_add.non_space_chars;
}

