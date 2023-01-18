////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/lock_registration/detail/register_locks.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/thread_support/spinlock.hpp>

#include <utility>

namespace hpx::util {

    // Lockable spinlock class
    struct spinlock
    {
    public:
        HPX_NON_COPYABLE(spinlock);

    private:
        hpx::util::detail::spinlock m;

    public:
        spinlock() noexcept
        {
            HPX_ITT_SYNC_CREATE(this, "util::spinlock", nullptr);
        }

        explicit spinlock(char const* desc) noexcept
        {
            HPX_ITT_SYNC_CREATE(this, "util::spinlock", desc);
        }

        ~spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock() noexcept(
            noexcept(util::register_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_PREPARE(this);
            m.lock();
            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock() noexcept(
            noexcept(util::register_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_PREPARE(this);
            if (m.try_lock())
            {
                HPX_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return true;
            }
            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock() noexcept(
            noexcept(util::unregister_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_RELEASING(this);
            m.unlock();
            HPX_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }
    };
}    // namespace hpx::util
