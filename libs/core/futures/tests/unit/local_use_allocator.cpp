//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (C) 2011 Vicente J. Botet Escriba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>

#include <memory>

#include "test_allocator.hpp"

int main()
{
    // hpx::promise
    static_assert(
        std::uses_allocator<hpx::promise<int>, test_allocator<int>>::value,
        "std::uses_allocator<hpx::promise<int>, test_allocator<int> "
        ">::value");
    static_assert(
        std::uses_allocator<hpx::promise<int&>, test_allocator<int>>::value,
        "std::uses_allocator<hpx::promise<int&>, test_allocator<int> "
        ">::value");
    static_assert(
        std::uses_allocator<hpx::promise<void>, test_allocator<void>>::value,
        "std::uses_allocator<hpx::promise<void>, test_allocator<void> "
        ">::value");

    // hpx::lcos::local::packaged_task
    static_assert(std::uses_allocator<hpx::packaged_task<int()>,
                      test_allocator<int>>::value,
        "std::uses_allocator<hpx::packaged_task<int()>, "
        "test_allocator<int> >::value");

    return 0;
}
