//  Copyright (c) 2024 Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/concurrency/detail/lockfree_uint128_type.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <iostream>

void lockfree_128uint_test()
{
    hpx::lockfree::uint128_type x(1, 2);
    hpx::lockfree::uint128_type y(3, 4);
    std::atomic<hpx::lockfree::uint128_type> a(x);
    std::atomic<hpx::lockfree::uint128_type> b(y);

    a.is_lock_free() ? std::cout << "is_lock_free() passed" << std::endl :
                       std::cout << "is_lock_free() failed" << std::endl;

    hpx::lockfree::uint128_type expected = x;
    hpx::lockfree::uint128_type desired = y;
    bool result =
        a.compare_exchange_weak(expected, desired, std::memory_order_seq_cst);
    hpx::lockfree::uint128_type received = a.load();
    (result && (received == desired)) ?
        std::cout << "compare_exchange_weak() passed" << std::endl :
        std::cout << "compare_exchange_weak() failed" << std::endl;
}

int main()
{
    lockfree_128uint_test();
    return hpx::util::report_errors();
}
