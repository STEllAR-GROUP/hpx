////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2007 Douglas Gregor
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <tuple>

template <typename... Elements>
struct make_tuple
{
    typedef std::tuple<Elements...> type;
};

int main()
{
   make_tuple<int, int, int, long, float, make_tuple<bool>::type>::type t;
   return 0;
}

