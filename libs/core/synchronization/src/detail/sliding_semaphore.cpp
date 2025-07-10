//  Copyright (c) 2016-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/detail/sliding_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/assert_owns_lock.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
namespace hpx::lcos::local::detail {

    sliding_semaphore::sliding_semaphore(
        std::int64_t max_difference, std::int64_t lower_limit) noexcept
      : max_difference_(max_difference)
      , lower_limit_(lower_limit)
    {
    }

    sliding_semaphore::~sliding_semaphore() = default;

    void sliding_semaphore::set_max_difference(std::unique_lock<mutex_type>& l,
        std::int64_t max_difference, std::int64_t lower_limit) noexcept
    {
        HPX_ASSERT_OWNS_LOCK(l);

        max_difference_ = max_difference;
        lower_limit_ = lower_limit;
    }

    void sliding_semaphore::wait(
        std::unique_lock<mutex_type>& l, std::int64_t upper_limit)
    {
        HPX_ASSERT_OWNS_LOCK(l);

        while (upper_limit - max_difference_ > lower_limit_)    //-V776
        {
            cond_.wait(l, "sliding_semaphore::wait");
        }
    }

    bool sliding_semaphore::try_wait(
        std::unique_lock<mutex_type>& l, std::int64_t upper_limit)
    {
        HPX_ASSERT_OWNS_LOCK(l);

        if (upper_limit - max_difference_ <= lower_limit_)
        {
            // enter wait_locked only if necessary
            wait(l, upper_limit);
            return true;
        }
        return false;
    }

    void sliding_semaphore::signal(
        std::unique_lock<mutex_type> l, std::int64_t lower_limit)
    {
        HPX_ASSERT_OWNS_LOCK(l);

        mutex_type* mtx = l.mutex();

        lower_limit_ = (std::max) (lower_limit, lower_limit_);

        // touch upon all threads
        std::int64_t count = static_cast<std::int64_t>(cond_.size(l));
        for (/**/; count > 0; --count)
        {
            // notify_one() returns false if no more threads are waiting
            if (!cond_.notify_one(HPX_MOVE(l)))
                break;

            l = std::unique_lock<mutex_type>(*mtx);
        }
    }

    std::int64_t sliding_semaphore::signal_all(std::unique_lock<mutex_type> l)
    {
        HPX_ASSERT_OWNS_LOCK(l);

        signal(HPX_MOVE(l), lower_limit_);
        return lower_limit_;
    }
}    // namespace hpx::lcos::local::detail
