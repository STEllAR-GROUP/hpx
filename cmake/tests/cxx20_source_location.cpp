//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::source_location (C++20)

#include <iostream>
#include <source_location>

int main()
{
    std::cout << "file: " << location.file_name() << "(" << location.line()
              << ":" << location.column() << ") `" << location.function_name()
              << '\n';

    return 0;
}
