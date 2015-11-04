////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <initializer_list>
#include <vector>

template <typename T>
struct A
{
    std::vector<T> v_;

    A(std::initializer_list<T> v) : v_(v) {}
};

int main()
{
    A<int> a = {1, 2, 3, 4, 5};
}
