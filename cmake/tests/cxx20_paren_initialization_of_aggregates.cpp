//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of Parenthesized initialization of aggregates (C++20)

struct S
{
    int x;
    float y;
};

int main()
{
    S s(1, 2.0f);
    (void)s.x;
    (void)s.y;
    return 0;
}
