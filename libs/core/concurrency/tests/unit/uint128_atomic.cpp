//  Copyright (c) 2024 Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/concurrency/detail/uint128_atomic.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <iostream>

void uint128_atomic_test()
{
    hpx::lockfree::uint128_type x(1, 2);
    hpx::lockfree::uint128_type y(3, 4);
    hpx::lockfree::uint128_atomic a(x);
    hpx::lockfree::uint128_atomic b(y);

    HPX_TEST(a.is_lock_free());

    hpx::lockfree::uint128_type expected = x;
    hpx::lockfree::uint128_type desired = y;
    bool result =
        a.compare_exchange_weak(expected, desired, std::memory_order_seq_cst);
    hpx::lockfree::uint128_type received = a.load();
    HPX_TEST(result && (received == desired));
}

int main()
{
    uint128_atomic_test();
    return hpx::util::report_errors();
}
