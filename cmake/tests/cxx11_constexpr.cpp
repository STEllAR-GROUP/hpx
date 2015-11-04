////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Beman Dawes
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

constexpr int square(int x) { return x * x; }  // from N2235

struct A
{
   constexpr A(int i) : val(i) { }
private:
   int val;
};

constexpr A a = 42;

int main()
{
    constexpr int i = square(5);
}
