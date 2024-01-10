//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iterator>

struct A
{
    constexpr A() noexcept {}
    constexpr ~A() {}
};

int main()
{
    [[maybe_unused]] A a;
    return 0;
}
