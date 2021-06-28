//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of perfect lambda-capture for variadic arguments:
// [... ts = std::forward<Ts>(ts)] (C++20)

#include <utility>

template <typename... Ts>
void foo(Ts&&... ts)
{
    return [... ts = std::forward<Ts>(ts)]() {};
}

int main()
{
    foo(1, 2, 3)();
    return 0;
}
