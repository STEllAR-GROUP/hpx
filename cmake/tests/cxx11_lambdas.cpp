////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009 Beman Dawes
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template <class Func>
int f(Func f) { return f(); }

int main()
{
    f([](){ return 0; });
}
