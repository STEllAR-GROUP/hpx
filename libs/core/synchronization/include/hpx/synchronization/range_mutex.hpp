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
        std::reference_wrapper<RangeMutex> mutex_ref;
        std::size_t lock_id = 0;

    public:
        range_guard(RangeMutex& lock, std::size_t begin, std::size_t end)
          : mutex_ref(lock)
        {
            lock_id = mutex_ref.get().lock(begin, end);
        }
        ~range_guard()
        {
            mutex_ref.get().unlock(lock_id);
        }

        range_guard(range_guard<RangeMutex>&) = delete;
        range_guard<RangeMutex>& operator=(range_guard<RangeMutex>&) = delete;

        range_guard(range_guard<RangeMutex>&& rhs_lock)
        {
            mutex_ref.get().unlock(lock_id);
            mutex_ref = rhs_lock.mutex_ref;
            lock_id = rhs_lock.lock_id;
            rhs_lock.lock_id = 0;
        }

        range_guard<RangeMutex>& operator=(range_guard<RangeMutex>&& rhs_lock)
        {
            mutex_ref.get().unlock(lock_id);
            mutex_ref = rhs_lock.mutex_ref;
            lock_id = rhs_lock.lock_id;
            rhs_lock.lock_id = 0;    // invalidating rhs_lock
        }
    };

    template <typename RangeMutex>
    class range_unique_lock
    {
        std::reference_wrapper<RangeMutex> mutex_ref;
        std::size_t lock_id = 0;

    public:
        range_unique_lock(RangeMutex& lock, std::size_t begin, std::size_t end)
          : mutex_ref(lock)
        {
            lock_id = mutex_ref.get().lock(begin, end);
        }

        ~range_unique_lock()
        {
            mutex_ref.get().unlock(lock_id);
        }

        range_unique_lock(range_guard<RangeMutex>&) = delete;
        range_unique_lock<RangeMutex>& operator=(
            range_unique_lock<RangeMutex>) = delete;

        range_unique_lock<RangeMutex>& operator=(
            range_unique_lock<RangeMutex>&& rhs_lock)
        {
            mutex_ref.get().unlock(lock_id);
            mutex_ref = rhs_lock.mutex_ref;
            lock_id = rhs_lock.lock_id;
            rhs_lock.lock_id = 0;    // invalidating rhs_lock
        }

        void lock(std::size_t begin, std::size_t end)
        {
            lock_id = mutex_ref.get().lock(begin, end);
        }

        void try_lock(std::size_t begin, std::size_t end)
        {
            lock_id = mutex_ref.get().try_lock(begin, end);
        }

        void unlock()
        {
            mutex_ref.get().unlock(lock_id);
            lock_id = 0;
        }

        void swap(range_unique_lock<RangeMutex>& uLock)
        {
            std::swap(mutex_ref, uLock.mutex_ref);
            std::swap(lock_id, uLock.lock_id);
        }

        RangeMutex* release()
        {
            RangeMutex* mtx = mutex_ref.get();
            mutex_ref = nullptr;
            lock_id = 0;
            return mtx;
        }

        operator bool() const
        {
            return lock_id != 0;
        }

        bool owns_lock() const
        {
            return lock_id != 0;
        }

        RangeMutex* mutex() const
        {
            return mutex_ref.get();
        }
    };
}    // namespace hpx::synchronization
