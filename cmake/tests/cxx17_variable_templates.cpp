//  Copyright (C) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

struct true_type
{
    enum { value = 1; }
}

template <typename T>
struct is_foobar : true_type {};

template <typename T>
constexpr bool is_foobar_v = is_foobar<T>::value;

int main() {}
