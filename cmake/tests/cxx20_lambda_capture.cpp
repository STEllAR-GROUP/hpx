//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of Allow lambda-capture [=, this] (C++20)

struct S
{
    void f()
    {
        (void)[=, this]{};
    };
};

int main()
{
    return 0;
}
