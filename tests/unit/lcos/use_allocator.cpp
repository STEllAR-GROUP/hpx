//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (C) 2011 Vicente J. Botet Escriba
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <memory>

#include "test_allocator.hpp"

int main()
{
    // hpx::lcos::local::promise
    static_assert(
        std::uses_allocator<
            hpx::lcos::local::promise<int>, test_allocator<int>
        >::value,
        "std::uses_allocator<local::promise<int>, test_allocator<int> >::value");
    static_assert(
        std::uses_allocator<
            hpx::lcos::local::promise<int&>, test_allocator<int>
        >::value,
        "std::uses_allocator<local::promise<int&>, test_allocator<int> >::value");
    static_assert(
        std::uses_allocator<
            hpx::lcos::local::promise<void>, test_allocator<void>
        >::value,
        "std::uses_allocator<local::promise<void>, test_allocator<void> >::value");

    // hpx::lcos::promise
    static_assert(
        std::uses_allocator<
            hpx::lcos::promise<int>, test_allocator<int>
        >::value,
        "std::uses_allocator<promise<int>, test_allocator<int> >::value");
    static_assert(
        std::uses_allocator<
            hpx::lcos::promise<void>, test_allocator<void>
        >::value,
        "std::uses_allocator<promise<void>, test_allocator<void> >::value");

    // hpx::lcos::local::packaged_task
    static_assert(
        std::uses_allocator<
            hpx::lcos::local::packaged_task<int()>, test_allocator<int>
        >::value,
        "std::uses_allocator<local::packaged_task<int()>, "
            "test_allocator<int> >::value");

    return hpx::util::report_errors();
}
