#pragma once

#include <hpx/execution_base/this_thread.hpp>

#include <atomic>
#include <boost/container/flat_map.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace hpx::synchronization::detail {

    template <typename Lock, template <typename> typename Guard>
    class RangeLock
    {
        template <typename Key, typename Value>
        using MapTy = boost::container::flat_map<Key, Value>;

        Lock mtx;
        std::size_t counter = 0;
        MapTy<std::size_t, std::pair<std::size_t, std::size_t>> rangeMap;
        MapTy<std::size_t, std::shared_ptr<std::atomic_bool>> waiting;

    public:
        std::size_t lock(std::size_t begin, std::size_t end);
        std::size_t try_lock(std::size_t begin, std::size_t end);
        void unlock(std::size_t lockId);
    };

    template <class Lock, template <class> class Guard>
    std::size_t RangeLock<Lock, Guard>::lock(std::size_t begin, std::size_t end)
    {
        std::size_t lockId = 0;
        bool localFlag = false;
        std::size_t blockIdx;

        std::shared_ptr<std::atomic_bool> waitingFlag;

        while (lockId == 0)
        {
            {
                const Guard<Lock> lock_guard(mtx);
                for (auto const& it : rangeMap)
                {
                    std::size_t b = it.second.first;
                    std::size_t e = it.second.second;

                    if ((!(e < begin)) & (!(end < b)))
                    {
                        blockIdx = it.first;
                        localFlag = true;
                        waitingFlag = waiting[blockIdx];
                        break;
                    }
                }
                if (localFlag == false)
                {
                    ++counter;
                    rangeMap[counter] = {begin, end};
                    waiting[counter] = std::shared_ptr<std::atomic_bool>(
                        new std::atomic_bool(false));
                    lockId = counter;    // to get rid of codacy warning
                    return counter;
                }
                localFlag = false;
            }
            auto pred = [&waitingFlag]() noexcept {
                return waitingFlag->load();
            };
            util::yield_while<true>(pred, "hpx::range_lock::lock");
        }
        return lockId;    // should not reach here
    }

    template <class Lock, template <class> class Guard>
    void RangeLock<Lock, Guard>::unlock(std::size_t lockId)
    {
        const Guard lock_guard(mtx);

        rangeMap.erase(lockId);

        waiting[lockId]->store(true);

        waiting.erase(lockId);
        return;
    }

    template <class Lock, template <class> class Guard>
    std::size_t RangeLock<Lock, Guard>::try_lock(
        std::size_t begin, std::size_t end)
    {
        const Guard lock_guard(mtx);
        for (auto const& it : rangeMap)
        {
            std::size_t b = it.second.first;
            std::size_t e = it.second.second;

            if (!(e < begin) && !(end < b))
            {
                return 0;
            }
        }
        rangeMap[++counter] = {begin, end};
        return counter;
    }
}    // namespace hpx::synchronization::detail
