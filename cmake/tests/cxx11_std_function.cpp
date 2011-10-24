////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <functional>

struct functor
{
    int operator()(int i, int j) const
    {
        return i + j;
    }
};

int free_function(int i, int j)
{
    return i + j;
}

int main()
{ 
    std::function<int(int, int)> f0(functor()), f1(free_function);

    return (f0() == 17) && (f1() == 17);
}

