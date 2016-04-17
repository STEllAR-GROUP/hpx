//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_MUTEX_HPP
#define HPX_LCOS_MUTEX_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/date_time_chrono.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    class mutex
    {
        HPX_NON_COPYABLE(mutex);

    protected:
        typedef lcos::local::spinlock mutex_type;

    public:
        HPX_EXPORT mutex(char const* const description = "");

        HPX_EXPORT ~mutex();

        HPX_EXPORT void lock(char const* description, error_code& ec = throws);

        void lock(error_code& ec = throws)
        {
            return lock("mutex::lock", ec);
        }

        HPX_EXPORT bool try_lock(char const* description, error_code& ec = throws);

        bool try_lock(error_code& ec = throws)
        {
            return try_lock("mutex::try_lock", ec);
        }

        HPX_EXPORT void unlock(error_code& ec = throws);

    protected:
        mutable mutex_type mtx_;
        threads::thread_id_repr_type owner_id_;
        detail::condition_variable cond_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class timed_mutex : private mutex
    {
        HPX_NON_COPYABLE(timed_mutex);

    public:
        HPX_EXPORT timed_mutex(char const* const description = "");

        HPX_EXPORT ~timed_mutex();

        using mutex::lock;
        using mutex::try_lock;
        using mutex::unlock;

        HPX_EXPORT bool try_lock_until(util::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        bool try_lock_until(util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return try_lock_until(abs_time, "mutex::try_lock_until", ec);
        }

        bool try_lock_for(util::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return try_lock_until(rel_time.from_now(), description, ec);
        }

        bool try_lock_for(util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return try_lock_for(rel_time, "mutex::try_lock_for", ec);
        }
    };
}}}

#endif
