////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <boost/array.hpp>

int main()
{
    boost::array<unsigned, 10> fib = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 };

    unsigned total = 0;

    std::for_each(fib.begin(), fib.end(),
        [&total](int x) { total += x; }
    );

    return !(88 == total);
}

