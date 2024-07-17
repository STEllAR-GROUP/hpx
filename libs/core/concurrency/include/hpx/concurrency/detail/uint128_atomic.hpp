//  Copyright (c) 2024      Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concurrency/detail/uint128_type.hpp>
#include <cstddef>

#include <atomic>

/* Note: currently only supported on Linux */
#if !defined(HPX_WITH_CXX11_ATOMIC_128BIT_LOCKFREE) && defined(__linux__)

namespace hpx::lockfree {
    struct uint128_atomic : std::atomic<hpx::lockfree::uint128_type>
    {
        constexpr uint128_atomic(std::size_t l, std::size_t r) noexcept
          : std::atomic<hpx::lockfree::uint128_type>(uint128_type{l, r})
        {
        }

        constexpr uint128_atomic(hpx::lockfree::uint128_type value) noexcept
          : std::atomic<hpx::lockfree::uint128_type>(value)
        {
        }

        bool is_lock_free() const noexcept
        {
            return true;
        }

        bool compare_exchange_weak(hpx::lockfree::uint128_type& expected,
            hpx::lockfree::uint128_type desired,
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

        bool compare_exchange_strong(hpx::lockfree::uint128_type& expected,
            hpx::lockfree::uint128_type desired,
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
    };
}    // namespace hpx::lockfree

#else

namespace hpx::lockfree {
    /* Alias std::atomic if custom implementation isn't supported */
    using uint128_atomic = std::atomic<uint128_type>;
}    // namespace hpx::lockfree
#endif    // !defined(HPX_WITH_CXX11_ATOMIC_128BIT_LOCKFREE) && defined(__linux__)
