////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <utility>

struct test_class1 { void foo() {} };
struct test_class2 { void bar() {} };

template <typename T>
decltype(std::declval<T>().foo()) foo()
{}

template <typename T>
decltype(std::declval<T>().bar()) foo()
{}

int main()
{
    foo<test_class1>();
    foo<test_class2>();
}
