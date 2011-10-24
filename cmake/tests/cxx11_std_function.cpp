////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <functional>

namespace boost
{
    using ::_1;
    using ::_2;
    using ::_3;
}

namespace std
{
    using ::std::placeholders::_1;
    using ::std::placeholders::_2;
    using ::std::placeholders::_3;
}

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

int take_reference(functor& f, int i, int j)
{
    return f(i, j);
}

int main()
{ 
    functor f;

    std::function<int(int, int)>
        f0(f)
      , f1(free_function)
      , f2(boost::bind(&functor::operator(), &f, boost::_1, boost::_2))
      , f3(std::bind(&functor::operator(), &f, std::_1, std::_2));

    std::function<int(functor&, int, int)>
        f4(boost::bind(&take_reference, boost::_1, boost::_2, boost::_3))
      , f5(std::bind(&take_reference, std::_1, std::_2, std::_3));

    return !(   (f0(9, 8) == 17)
            && (f1(15, 2) == 17)
            && (f2(15, 2) == 17)
            && (f3(3, 14) == 17)
            && (f4(f, 6, 11) == 17)
            && (f5(f, 1, 16) == 17));
}

