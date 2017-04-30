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
    HPX_TEST_EQ(test_alloc_base::count, 0);
    {
        hpx::lcos::local::promise<int> p(
            std::allocator_arg, test_allocator<int>());
        HPX_TEST_EQ(test_alloc_base::count, 1);
        hpx::future<int> f = p.get_future();
        HPX_TEST_EQ(test_alloc_base::count, 1);
        HPX_TEST(f.valid());
    }
    HPX_TEST_EQ(test_alloc_base::count, 0);
    {
        hpx::lcos::local::promise<int&> p(
            std::allocator_arg, test_allocator<int>());
        HPX_TEST_EQ(test_alloc_base::count, 1);
        hpx::future<int&> f = p.get_future();
        HPX_TEST_EQ(test_alloc_base::count, 1);
        HPX_TEST(f.valid());
    }
    HPX_TEST_EQ(test_alloc_base::count, 0);
    {
        hpx::lcos::local::promise<void> p(
            std::allocator_arg, test_allocator<void>());
        HPX_TEST_EQ(test_alloc_base::count, 1);
        hpx::future<void> f = p.get_future();
        HPX_TEST_EQ(test_alloc_base::count, 1);
        HPX_TEST(f.valid());
    }
    HPX_TEST_EQ(test_alloc_base::count, 0);

    return hpx::util::report_errors();
}

