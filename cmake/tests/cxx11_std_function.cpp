////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/bind.hpp>

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
    functor f;

    std::function<int(int, int)>
        f0(f)
      , f1(free_function)
      , f2(boost::bind(&functor::operator(), &f, _1, _2));

    return !(   (f0(9, 8) == 17)
             && (f1(15, 2) == 17)
             && (f2(10, 7) == 17));
}

