////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Beman Dawes
//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template <typename T = int>
T foo() { return 0; }

template <typename T, typename U>
bool is_same(T, U) { return false; }

template <typename T>
bool is_same(T, T) { return true; }

template <typename T = int>
T bar(T v = T()) { return v; }

int main()
{
    bool b = !is_same(foo<>(), 0) || is_same(foo<>(), 0L);
    bar(0) == bar();
}
