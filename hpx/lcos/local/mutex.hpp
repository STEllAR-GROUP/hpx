//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_MUTEX_JUN_23_2008_0530PM)
#define HPX_LCOS_MUTEX_JUN_23_2008_0530PM

#include <hpx/config.hpp>
#include <hpx/config/emulate_deleted.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/intrusive/slist.hpp>
#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    class mutex
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        typedef boost::unique_lock<mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<mutex> scoped_try_lock;

    public:
        mutex(char const* const description = "")
          : owner_id_(threads::invalid_thread_id_repr)
        {
            HPX_ITT_SYNC_CREATE(this, "lcos::local::mutex", description);
            HPX_ITT_SYNC_RENAME(this, "lcos::local::mutex");
        }

        HPX_NON_COPYABLE(mutex);

        ~mutex()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock(char const* description, error_code& ec = throws)
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

        void lock(error_code& ec = throws)
        {
            return lock("mutex::lock", ec);
        }

        bool try_lock(char const* description, error_code& ec = throws)
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

        bool try_lock(error_code& ec = throws)
        {
            return try_lock("mutex::try_lock", ec);
        }

        bool try_lock_until(util::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws)
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

                if (reason == threads::wait_signaled) //-V110
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

        void unlock(error_code& ec = throws)
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

    private:
        mutable mutex_type mtx_;
        threads::thread_id_repr_type owner_id_;
        detail::condition_variable cond_;
    };

    typedef mutex timed_mutex;

}}}

#endif
