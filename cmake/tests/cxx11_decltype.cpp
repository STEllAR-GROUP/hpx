////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

int foo()
{
    return 0;
}

decltype((42)) foo2()
{
    return 42;
}

template <typename F>
void bar(F f)
{
    decltype(f()) i = 42;
}

int main()
{
    bar(&foo);
    bar(&foo2);
}
