//  Copyright (c) 2024      Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concurrency/detail/uint128_type.hpp>

#include <atomic>

#if !defined(HPX_WITH_CXX11_ATOMIC_128BIT_LOCKFREE)
/* this will only be true if the target supports cmpxchg16b */
template <>
inline bool std::atomic<hpx::lockfree::uint128_type>::is_lock_free()
    const noexcept
{
    return true;
}

template <>
inline bool std::atomic<hpx::lockfree::uint128_type>::compare_exchange_weak(
    hpx::lockfree::uint128_type& expected, hpx::lockfree::uint128_type desired,
    std::memory_order memOrder) noexcept
{
    (void) memOrder;
    bool result = false;
    asm("lock cmpxchg16b %1\n\t"
        "setz %0"
        : "=r"(result),
        "+m"(*reinterpret_cast<hpx::lockfree::uint128_type*>(this)),
        "+d"(expected.right),    // high
        "+a"(expected.left)      // low
        : "c"(desired.right),    // high
        "b"(desired.left)        // low
        : "cc", "memory");
    return result;
}

template <>
inline bool std::atomic<hpx::lockfree::uint128_type>::compare_exchange_strong(
    hpx::lockfree::uint128_type& expected, hpx::lockfree::uint128_type desired,
    std::memory_order memOrder) noexcept
{
    (void) memOrder;
    bool result = false;
    asm("lock cmpxchg16b %1\n\t"
        "setz %0"
        : "=r"(result),
        "+m"(*reinterpret_cast<hpx::lockfree::uint128_type*>(this)),
        "+d"(expected.right),    // high
        "+a"(expected.left)      // low
        : "c"(desired.right),    // high
        "b"(desired.left)        // low
        : "cc", "memory");
    return result;
}
#endif    // !defined(HPX_WITH_CXX11_ATOMIC_128BIT_LOCKFREE)
