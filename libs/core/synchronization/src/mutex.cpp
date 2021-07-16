//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/synchronization/mutex.hpp>

#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local {
    ///////////////////////////////////////////////////////////////////////////
    mutex::mutex(char const* const description)
      : owner_id_(threads::invalid_thread_id)
    {
        HPX_ITT_SYNC_CREATE(this, "lcos::local::mutex", description);
        HPX_ITT_SYNC_RENAME(this, "lcos::local::mutex");
    }

    mutex::~mutex()
    {
        HPX_ITT_SYNC_DESTROY(this);
    }

    void mutex::lock(char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_type self_id = threads::get_self_id();
        if (owner_id_ == self_id)
        {
            HPX_ITT_SYNC_CANCEL(this);
            l.unlock();
            HPX_THROWS_IF(ec, deadlock, description,
                "The calling thread already owns the mutex");
            return;
        }

        while (owner_id_ != threads::invalid_thread_id)
        {
            cond_.wait(l, ec);
            if (ec)
            {
                HPX_ITT_SYNC_CANCEL(this);
                return;
            }
        }

        util::register_lock(this);
        HPX_ITT_SYNC_ACQUIRED(this);
        owner_id_ = self_id;
    }

    bool mutex::try_lock(char const* /* description */, error_code& /* ec */)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
        std::unique_lock<mutex_type> l(mtx_);

        if (owner_id_ != threads::invalid_thread_id)
        {
            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        threads::thread_id_type self_id = threads::get_self_id();
        util::register_lock(this);
        HPX_ITT_SYNC_ACQUIRED(this);
        owner_id_ = self_id;
        return true;
    }

    void mutex::unlock(error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_RELEASING(this);
        // Unregister lock early as the lock guard below may suspend.
        util::unregister_lock(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_type self_id = threads::get_self_id();
        if (HPX_UNLIKELY(owner_id_ != self_id))
        {
            l.unlock();
            HPX_THROWS_IF(ec, lock_error, "mutex::unlock",
                "The calling thread does not own the mutex");
            return;
        }

        HPX_ITT_SYNC_RELEASED(this);
        owner_id_ = threads::invalid_thread_id;

        {
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);
            cond_.notify_one(std::move(l), threads::thread_priority::boost, ec);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    timed_mutex::timed_mutex(char const* const description)
      : mutex(description)
    {
    }

    timed_mutex::~timed_mutex() {}

    bool timed_mutex::try_lock_until(
        hpx::chrono::steady_time_point const& abs_time,
        char const* /* description */, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_type self_id = threads::get_self_id();
        if (owner_id_ != threads::invalid_thread_id)
        {
            threads::thread_restart_state const reason =
                cond_.wait_until(l, abs_time, ec);
            if (ec)
            {
                HPX_ITT_SYNC_CANCEL(this);
                return false;
            }

            if (reason == threads::thread_restart_state::timeout)    //-V110
            {
                HPX_ITT_SYNC_CANCEL(this);
                return false;
            }

            if (owner_id_ != threads::invalid_thread_id)    //-V110
            {
                HPX_ITT_SYNC_CANCEL(this);
                return false;
            }
        }

        util::register_lock(this);
        HPX_ITT_SYNC_ACQUIRED(this);
        owner_id_ = self_id;
        return true;
    }
}}}    // namespace hpx::lcos::local
