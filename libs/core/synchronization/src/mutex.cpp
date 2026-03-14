//  Copyright (c) 2007-2026 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/modules/coroutines.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/modules/tracy.hpp>
#endif

#include <mutex>
#include <string>
#include <utility>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
#if HPX_HAVE_ITTNOTIFY != 0 || defined(HPX_HAVE_MODULE_TRACY)
    mutex::mutex(char const* const description)
      : owner_id_(threads::invalid_thread_id)
    {
        HPX_ITT_SYNC_CREATE(this, "hpx::mutex", description);
#if defined(HPX_HAVE_MODULE_TRACY)
        context_ = hpx::tracy::create(std::string("hpx::mutex") + description);
#endif
    }
#endif

#if HPX_HAVE_ITTNOTIFY != 0 || defined(HPX_HAVE_MODULE_TRACY)
    mutex::~mutex()
    {
        HPX_ITT_SYNC_DESTROY(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::destroy(context_);
#endif
    }
#else
    mutex::~mutex() = default;
#endif

    void mutex::lock(char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        bool const run_after = hpx::tracy::lock_prepare(context_);
#endif

        {
            std::unique_lock<mutex_type> l(mtx_);

            threads::thread_id_type const self_id = threads::get_self_id();
            if (owner_id_ == self_id)
            {
                HPX_ITT_SYNC_CANCEL(this);
                l.unlock();
                HPX_THROWS_IF(ec, hpx::error::deadlock, description,
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
            owner_id_ = self_id;
        }

        HPX_ITT_SYNC_ACQUIRED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        if (run_after)
            hpx::tracy::lock_acquired(context_);
#endif
    }

    bool mutex::try_lock(char const* /* description */, error_code& /* ec */)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        bool const run_after = hpx::tracy::lock_prepare(context_);
#endif

        {
            std::unique_lock<mutex_type> l(mtx_);

            if (owner_id_ != threads::invalid_thread_id)
            {
                HPX_ITT_SYNC_CANCEL(this);
#if defined(HPX_HAVE_MODULE_TRACY)
                if (run_after)
                    hpx::tracy::lock_acquired(context_, false);
#endif
                return false;
            }

            util::register_lock(this);
            owner_id_ = threads::get_self_id();
        }

        HPX_ITT_SYNC_ACQUIRED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        if (run_after)
            hpx::tracy::lock_acquired(context_, true);
#endif

        return true;
    }

    void mutex::unlock(error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_RELEASING(this);
        // Unregister lock early as the lock guard below may suspend.
        util::unregister_lock(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_type const self_id = threads::get_self_id();
        if (HPX_UNLIKELY(owner_id_ != self_id))
        {
            l.unlock();
            HPX_THROWS_IF(ec, hpx::error::lock_error, "mutex::unlock",
                "The calling thread does not own the mutex");
            return;
        }

        owner_id_ = threads::invalid_thread_id;

        HPX_ITT_SYNC_RELEASED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::lock_released(context_);
#endif

        {
            [[maybe_unused]] util::ignore_while_checking il(&l);

            // Failing to release lock 'this->mtx' in function
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26115)
#endif

            cond_.notify_one(HPX_MOVE(l), threads::thread_priority::boost, ec);
            il.reset_owns_registration();

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    timed_mutex::timed_mutex(char const* const description)
      : mutex(description)
    {
    }

    timed_mutex::~timed_mutex() = default;

    bool timed_mutex::try_lock_until(
        hpx::chrono::steady_time_point const& abs_time,
        char const* /* description */, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != nullptr);

        HPX_ITT_SYNC_PREPARE(this);
        std::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_type const self_id = threads::get_self_id();
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
}    // namespace hpx
