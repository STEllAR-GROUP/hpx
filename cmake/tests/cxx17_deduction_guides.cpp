//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

template <typename T>
struct test
{
    template <typename T_>
    test(T_&&)
    {
    }
};

template <typename T>
test(T) -> test<T>;

int main()
{
    test t{42};

    return 0;
}
