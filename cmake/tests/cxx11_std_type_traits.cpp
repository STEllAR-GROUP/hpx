////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <type_traits>

struct callable
{
    int operator()(){ return 0; }
};

int main()
{
    using namespace std;

    int x = 0;
    add_const<int>::type* rc = &x;
    decay<int const&>::type* d = &x;
    result_of<callable()>::type* ro = &x;
    is_convertible<int, long>::type ic;
}
