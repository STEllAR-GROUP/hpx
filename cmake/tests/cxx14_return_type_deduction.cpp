////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

auto check(int a, int b)
{
    int c = a + b;
    return c;
}

template <typename T>
auto check2(T num);

template <typename T>
auto check2(T num)
{
    return num * num;
}

int main()
{
    int a = 10;
    int b = 20;
    check(a, b);
    double c = check2(30.0);
}

