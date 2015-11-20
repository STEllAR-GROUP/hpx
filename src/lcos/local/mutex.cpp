//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/local/mutex.hpp>

#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    mutex::mutex(char const* const description)
      : owner_id_(threads::invalid_thread_id_repr)
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
        HPX_ASSERT(threads::get_self_ptr() != 0);

        HPX_ITT_SYNC_PREPARE(this);
        boost::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_repr_type self_id = threads::get_self_id().get();
        if(owner_id_ == self_id)
        {
            HPX_ITT_SYNC_CANCEL(this);
            HPX_THROWS_IF(ec, deadlock,
                description,
                "The calling thread already owns the mutex");
            return;
        }

        while (owner_id_ != threads::invalid_thread_id_repr)
        {
            cond_.wait(l, ec);
            if (ec) { HPX_ITT_SYNC_CANCEL(this); return; }
        }

        util::register_lock(this);
        HPX_ITT_SYNC_ACQUIRED(this);
        owner_id_ = self_id;
    }

    bool mutex::try_lock(char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != 0);

        HPX_ITT_SYNC_PREPARE(this);
        boost::unique_lock<mutex_type> l(mtx_);

        if (owner_id_ != threads::invalid_thread_id_repr)
        {
            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        threads::thread_id_repr_type self_id = threads::get_self_id().get();
        util::register_lock(this);
        HPX_ITT_SYNC_ACQUIRED(this);
        owner_id_ = self_id;
        return true;
    }

    void mutex::unlock(error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != 0);

        HPX_ITT_SYNC_RELEASING(this);
        boost::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_repr_type self_id = threads::get_self_id().get();
        if (HPX_UNLIKELY(owner_id_ != self_id))
        {
            util::unregister_lock(this);
            HPX_THROWS_IF(ec, lock_error,
                "mutex::unlock",
                "The calling thread does not own the mutex");
            return;
        }

        util::unregister_lock(this);
        HPX_ITT_SYNC_RELEASED(this);
        owner_id_ = threads::invalid_thread_id_repr;

        cond_.notify_one(std::move(l), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    timed_mutex::timed_mutex(char const* const description)
      : mutex(description)
    {}

    timed_mutex::~timed_mutex()
    {}

    bool timed_mutex::try_lock_until(util::steady_time_point const& abs_time,
        char const* description, error_code& ec)
    {
        HPX_ASSERT(threads::get_self_ptr() != 0);

        HPX_ITT_SYNC_PREPARE(this);
        boost::unique_lock<mutex_type> l(mtx_);

        threads::thread_id_repr_type self_id = threads::get_self_id().get();
        if (owner_id_ != threads::invalid_thread_id_repr)
        {
            threads::thread_state_ex_enum const reason =
                cond_.wait_until(l, abs_time, ec);
            if (ec) { HPX_ITT_SYNC_CANCEL(this); return false; }

            if (reason == threads::wait_timeout) //-V110
            {
                HPX_ITT_SYNC_CANCEL(this);
                return false;
            }

            if (owner_id_ != threads::invalid_thread_id_repr) //-V110
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
}}}
