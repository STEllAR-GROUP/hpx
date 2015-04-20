////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

using PINT = void (*)(int);             // using plus C-style type

template <typename Arg>
using PFUN = void (*)(Arg);

void check_f(PINT) {}

int main()
{
    PFUN<int> x;
    check_f(x);
}
