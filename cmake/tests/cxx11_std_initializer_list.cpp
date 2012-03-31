////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <initializer_list>
 
template <
    typename T
>
struct A
{
    std::vector<T> v_;

    A(std::initializer_list<T> v) : v_(v) {}
};
 
int main()
{
    A<int> a = {1, 2, 3, 4, 5};

    return !(  (5 == a.v_.size())
            && (1 == a.v_[0])
            && (2 == a.v_[1])
            && (3 == a.v_[2])
            && (4 == a.v_[3])
            && (5 == a.v_[4]));
}

