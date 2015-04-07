////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Kohei Takahashi
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

template <class Func>
int f(Func f) { return f(0); }

int main()
{
    f([x = 0](auto ret) { return ret; });
}
