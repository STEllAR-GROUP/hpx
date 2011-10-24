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

    int operator()(int i, int j, int k) const
    {
        return i + j + k;
    }
};

int free_function(int i, int j)
{
    return i + j;
}

int main()
{ 
    using std::placeholders::_1;
    using std::placeholders::_2;

    functor f;

    return !(   (std::bind(&functor::operator(), &f, _1, _2)(16, 1) == 17) 
             && (std::bind(f, _1, 9)(8) == 17)
             && (std::bind(&free_function, 5, _2)(12, 8) == 17));
}

