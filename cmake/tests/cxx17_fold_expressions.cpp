////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template <class... Args>
void foo(Args&&... args)
{
    auto a = (... + args);
    auto b = (args + ...);
    auto c = (1 + ... + args);
    auto d = (args + ... + 1);
}

int main()
{
    foo(1, 2.f, 3.);
}
