//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/timing/steady_clock.hpp>

namespace hpx { namespace threads {

    using thread_id_type = thread_id;
    using thread_self = coroutines::detail::coroutine_self;

    /// The function \a get_self_id returns the HPX thread id of the current
    /// thread (or zero if the current thread is not a HPX thread).
    HPX_CORE_EXPORT thread_id_type get_self_id();

    /// The function \a get_self_ptr returns a pointer to the (OS thread
    /// specific) self reference to the current HPX thread.
    HPX_CORE_EXPORT thread_self* get_self_ptr();

}}    // namespace hpx::threads

namespace hpx { namespace lcos { namespace local {
    ///////////////////////////////////////////////////////////////////////////
    class mutex
    {
    public:
        HPX_NON_COPYABLE(mutex);

    protected:
        typedef lcos::local::spinlock mutex_type;

    public:
        HPX_CORE_EXPORT mutex(char const* const description = "");

        HPX_CORE_EXPORT ~mutex();

        HPX_CORE_EXPORT void lock(
            char const* description, error_code& ec = throws);

        void lock(error_code& ec = throws)
        {
            return lock("mutex::lock", ec);
        }

        HPX_CORE_EXPORT bool try_lock(
            char const* description, error_code& ec = throws);

        bool try_lock(error_code& ec = throws)
        {
            return try_lock("mutex::try_lock", ec);
        }

        HPX_CORE_EXPORT void unlock(error_code& ec = throws);

    protected:
        mutable mutex_type mtx_;
        threads::thread_id_type owner_id_;
        lcos::local::detail::condition_variable cond_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class timed_mutex : private mutex
    {
    public:
        HPX_NON_COPYABLE(timed_mutex);

    public:
        HPX_CORE_EXPORT timed_mutex(char const* const description = "");

        HPX_CORE_EXPORT ~timed_mutex();

        using mutex::lock;
        using mutex::try_lock;
        using mutex::unlock;

        HPX_CORE_EXPORT bool try_lock_until(
            hpx::chrono::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        bool try_lock_until(hpx::chrono::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return try_lock_until(abs_time, "mutex::try_lock_until", ec);
        }

        bool try_lock_for(hpx::chrono::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return try_lock_until(rel_time.from_now(), description, ec);
        }

        bool try_lock_for(hpx::chrono::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return try_lock_for(rel_time, "mutex::try_lock_for", ec);
        }
    };
}}}    // namespace hpx::lcos::local
