//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// test for availability of hpx::construct_at (C++ 20)

#include <memory>

struct A
{
    A(int) {}
};

int main()
{
    unsigned char buffer[sizeof(A)];

    A* ptr = hpx::construct_at(reinterpret_cast<A*>(buffer), 42);
    std::destroy_at(ptr);

    return 0;
}
