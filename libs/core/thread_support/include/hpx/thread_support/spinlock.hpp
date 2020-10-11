////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Peter Dimov
//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <atomic>

namespace hpx { namespace util { namespace detail {

    /// Lockable spinlock class
    struct spinlock
    {
    public:
        HPX_NON_COPYABLE(spinlock);

    private:
        std::atomic_flag m = ATOMIC_FLAG_INIT;

        HPX_CORE_EXPORT void yield_k(unsigned) noexcept;

    public:
        spinlock() = default;

        bool try_lock() noexcept
        {
            return !m.test_and_set(std::memory_order_acquire);
        }

        void lock() noexcept
        {
            for (unsigned k = 0; !try_lock(); ++k)
                yield_k(k);
        }

        void unlock() noexcept
        {
            m.clear(std::memory_order_release);
        }
    };

}}}    // namespace hpx::util::detail
