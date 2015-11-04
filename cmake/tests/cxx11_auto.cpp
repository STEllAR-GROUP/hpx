////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009 Andrey Semashev
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

void check_f(int&) {}

int main()
{
    auto x = 10;
    check_f(x);
}
