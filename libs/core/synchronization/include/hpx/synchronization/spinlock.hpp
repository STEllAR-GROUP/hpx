////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2020 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Patrick Diehl
//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/local/config.hpp>

#include <hpx/execution_base/register_locks.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/modules/itt_notify.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    // std::mutex-compatible spinlock class
    struct spinlock
    {
    public:
        HPX_NON_COPYABLE(spinlock);

    private:
        std::atomic<bool> v_;

    public:
        spinlock(char const* const desc = "hpx::lcos::local::spinlock")
          : v_(false)
        {
            HPX_ITT_SYNC_CREATE(this, desc, "");
        }

        ~spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            // Checking for the value in is_locked() ensures that
            // acquire_lock is only called when is_locked computes
            // to false. This way we spin only on a load operation
            // which minimizes false sharing that comes with an
            // exchange operation.
            // Consider the following cases:
            // 1. Only one thread wants access critical section:
            //      is_locked() -> false; computes acquire_lock()
            //      acquire_lock() -> false (new value set to true)
            //      Thread acquires the lock and moves to critical
            //      section.
            // 2. Two threads simultaneously access critical section:
            //      Thread 1: is_locked() || acquire_lock() -> false
            //      Thread 1 acquires the lock and moves to critical
            //      section.
            //      Thread 2: is_locked() -> true; execution enters
            //      inside while without computing acquire_lock().
            //      Thread 2 yields while is_locked() computes to
            //      false. Then it retries doing is_locked() -> false
            //      followed by an acquire_lock() operation.
            //      The above order can be changed arbitrarily but
            //      the nature of execution will still remain the
            //      same.
            do
            {
                util::yield_while([this] { return is_locked(); },
                    "hpx::lcos::local::spinlock::lock");
            } while (!acquire_lock());

            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            bool r = acquire_lock();    //-V707

            if (r)
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
        HPX_FORCEINLINE bool acquire_lock()
        {
            return !v_.exchange(true, std::memory_order_acquire);
        }

        // relinquish lock
        HPX_FORCEINLINE void relinquish_lock()
        {
            v_.store(false, std::memory_order_release);
        }

        HPX_FORCEINLINE bool is_locked() const
        {
            return v_.load(std::memory_order_relaxed);
        }
    };
}}}    // namespace hpx::lcos::local
