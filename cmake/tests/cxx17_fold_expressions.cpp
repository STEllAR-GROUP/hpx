////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template <class... Args>
void foo(Args&&... args)
{
    auto a = (... + args);
    (void) a;
    auto b = (args + ...);
    (void) b;
    auto c = (1 + ... + args);
    (void) c;
    auto d = (args + ... + 1);
    (void) c;
}

int main()
{
    foo(1, 2.f, 3.);
}
