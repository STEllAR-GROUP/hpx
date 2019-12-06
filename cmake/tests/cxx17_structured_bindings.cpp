////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Jan Melech
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

struct A
{
    int a;
    float b;
};

int main()
{
    A a{1, 0.5f};
    auto& [x, y] = a;
}
