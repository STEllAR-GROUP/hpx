////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Peter Dimov
//  Copyright (c) 2020 Agustin Berge
//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// see https://rigtorp.se/spinlock/

#pragma once

#include <hpx/config.hpp>

#include <atomic>

namespace hpx::util::detail {

    /// Lockable spinlock class
    struct spinlock
    {
    private:
        std::atomic<bool> m;

        HPX_CORE_EXPORT void yield_k(unsigned) noexcept;

    public:
        constexpr spinlock() noexcept
          : m(false)
        {
        }

        HPX_NON_COPYABLE(spinlock);

        ~spinlock() = default;

        HPX_FORCEINLINE bool try_lock() noexcept
        {
            // First do a relaxed load to check if lock is free in order to
            // prevent unnecessary cache misses if someone does
            // while(!try_lock())
            return !m.load(std::memory_order_relaxed) &&
                !m.exchange(true, std::memory_order_acquire);
        }

        HPX_FORCEINLINE void lock() noexcept
        {
            // Wait for lock to be released without generating cache misses
            // Similar implementation to hpx::spinlock
            unsigned k = 0;
            while (!try_lock())
            {
                yield_k(k++);
            }
        }

        HPX_FORCEINLINE void unlock() noexcept
        {
            m.store(false, std::memory_order_release);
        }
    };
}    // namespace hpx::util::detail
