//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::aligned_alloc (C++17)

#include <cstdlib>

int main()
{
    char* s = std::aligned_alloc(1024, 1024 * sizeof(char));

    std::free(s);
    return 0;
}
