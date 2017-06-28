////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009 Andrey Semashev
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template<typename T>
auto check_f(T& i)
{
    T j = 1 + j;
    return j;
}

int main()
{
    int x = 10;
    int y = check_f(x);
}
