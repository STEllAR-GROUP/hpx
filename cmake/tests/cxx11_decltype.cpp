////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

int foo()
{}

template <typename F>
void bar(F f)
{
    decltype(f()) i = 42;
}

int main()
{
    bar(&foo);
    bar([]() -> decltype(42) {return 42;});
}
