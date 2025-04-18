//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of std::generator (C++ 23)

#include <generator>

std::generator<char> letters(char first)
{
    for (;; co_yield first++)
        ;
}

int main()
{
    for (const char ch : letters('a'))
    {
    }
    return 0;
}
