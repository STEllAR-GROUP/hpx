#pragma once

#include <mutex>

#include <hpx/synchronization/detail/range_lock_impl.hpp>
#include <hpx/synchronization/spinlock.hpp>

namespace hpx::synchronization {
    using range_lock =
        hpx::synchronization::detail::RangeLock<hpx::spinlock, std::lock_guard>;
}

// Lock guards for range_lock
namespace hpx::synchronization {

    template <typename RangeLock>
    class range_guard
    {
        std::reference_wrapper<RangeLock> lockRef;
        std::size_t lockId = 0;

    public:
        range_guard(RangeLock& lock, std::size_t begin, std::size_t end)
          : lockRef(lock)
        {
            lockId = lockRef.get().lock(begin, end);
        }
        ~range_guard()
        {
            lockRef.get().unlock(lockId);
        }
    };

}    // namespace hpx::synchronization

namespace hpx::synchronization {

    template <typename RangeLock>
    class range_unique_lock
    {
        std::reference_wrapper<RangeLock> lockRef;
        std::size_t lockId = 0;

    public:
        range_unique_lock(RangeLock& lock, std::size_t begin, std::size_t end)
          : lockRef(lock)
        {
            lockId = lockRef.get().lock(begin, end);
        }

        ~range_unique_lock()
        {
            lockRef.get().unlock(lockId);
        }

        void operator=(range_unique_lock<RangeLock>&& lock)
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

        void swap(std::unique_lock<RangeLock>& uLock)
        {
            std::swap(lockRef, uLock.lockRef);
            std::swap(lockId, uLock.lockId);
        }

        RangeLock* release()
        {
            RangeLock* mtx = lockRef.get();
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

        RangeLock* mutex() const
        {
            return lockRef.get();
        }
    };

}    // namespace hpx::synchronization
