//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <memory_resource>

int main()
{
    int memory[10];
    std::pmr::monotonic_buffer_resource pool(memory, 10);
    std::pmr::polymorphic_allocator<int> allocator(&pool);

    (void) allocator;

    return 0;
}
