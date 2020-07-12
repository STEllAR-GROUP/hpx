////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2020 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/itt_notify.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    /// boost::mutex-compatible spinlock class
    struct spinlock_no_backoff
    {
    public:
        HPX_NON_COPYABLE(spinlock_no_backoff);

    private:
        std::atomic<bool> v_;

    public:
        spinlock_no_backoff()
          : v_(0)
        {
            HPX_ITT_SYNC_CREATE(
                this, "hpx::lcos::local::spinlock_no_backoff", "");
        }

        ~spinlock_no_backoff()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            while (!acquire_lock())
            {
                util::yield_while([this] { return is_locked(); },
                    "hpx::lcos::local::spinlock_no_backoff::lock", false);
            }

            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            bool r = acquire_lock();    //-V707

            if (r == 0)
            {
                HPX_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return true;
            }

            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock()
        {
            HPX_ITT_SYNC_RELEASING(this);

            relinquish_lock();

            HPX_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }

    private:
        // returns whether the mutex has been acquired
        bool acquire_lock()
        {
            return !v_.exchange(true, std::memory_order_acquire);
        }

        // relinquish lock
        void relinquish_lock()
        {
            v_.store(false, std::memory_order_release);
        }

        bool is_locked() const
        {
            return v_.load(std::memory_order_relaxed);
        }
    };
}}}    // namespace hpx::lcos::local
