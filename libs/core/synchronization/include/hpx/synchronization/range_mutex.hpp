//  Copyright (c) 2023 Johan511
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is based on https://github.com/Johan511/ByteLock

#pragma once

#include <hpx/synchronization/detail/range_mutex_impl.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>
#include <functional>
#include <mutex>
#include <utility>

namespace hpx::synchronization {
    using range_mutex = hpx::synchronization::detail::range_mutex<hpx::spinlock,
        std::lock_guard>;

    // Lock guards for range_mutex

    template <typename RangeMutex>
    class range_guard
    {
        std::reference_wrapper<RangeMutex> lockRef;
        std::size_t lockId = 0;

    public:
        range_guard(RangeMutex& lock, std::size_t begin, std::size_t end)
          : lockRef(lock)
        {
            lockId = lockRef.get().lock(begin, end);
        }
        ~range_guard()
        {
            lockRef.get().unlock(lockId);
        }
    };

    template <typename RangeMutex>
    class range_unique_lock
    {
        std::reference_wrapper<RangeMutex> lockRef;
        std::size_t lockId = 0;

    public:
        range_unique_lock(RangeMutex& lock, std::size_t begin, std::size_t end)
          : lockRef(lock)
        {
            lockId = lockRef.get().lock(begin, end);
        }

        ~range_unique_lock()
        {
            lockRef.get().unlock(lockId);
        }

        void operator=(range_unique_lock<RangeMutex>&& lock)
        {
            lockRef.get().unlock(lockId);
            lockRef = lock.lockRef;
            lockId = lock.lockRef.get().lock();
        }

        void lock(std::size_t begin, std::size_t end)
        {
            lockId = lockRef.get().lock(begin, end);
        }

        void try_lock(std::size_t begin, std::size_t end)
        {
            lockId = lockRef.get().try_lock(begin, end);
        }

        void unlock()
        {
            lockRef.get().unlock(lockId);
            lockId = 0;
        }

        void swap(range_unique_lock<RangeMutex>& uLock)
        {
            std::swap(lockRef, uLock.lockRef);
            std::swap(lockId, uLock.lockId);
        }

        RangeMutex* release()
        {
            RangeMutex* mtx = lockRef.get();
            lockRef = nullptr;
            lockId = 0;
            return mtx;
        }

        operator bool() const
        {
            return lockId != 0;
        }

        bool owns_lock() const
        {
            return lockId != 0;
        }

        RangeMutex* mutex() const
        {
            return lockRef.get();
        }
    };
}    // namespace hpx::synchronization
