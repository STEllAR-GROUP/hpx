//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_MUTEX_JUN_23_2008_0530PM)
#define HPX_LCOS_MUTEX_JUN_23_2008_0530PM

#include <hpx/config.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/chrono/time_point.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    class mutex : boost::noncopyable
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(threads::thread_id_repr_type const& id)
              : id_(id)
            {}

            threads::thread_id_repr_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, queue_entry::hook_type,
            &queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_)
                    q_.erase(last_);     // remove entry from queue
            }

            queue_entry& e_;
            queue_type& q_;
            queue_type::const_iterator last_;
        };

        void abort_all()
        {
            while (!queue_.empty())
            {
                threads::thread_id_type id(
                    reinterpret_cast<threads::thread_data_base*>(queue_.front().id_));
                queue_.front().id_ = threads::invalid_thread_id_repr;
                queue_.pop_front();

                // we know that the id is actually the pointer to the thread
                LERR_(fatal)
                        << "mutex::abort_all:"
                        << " pending thread: "
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id << "): " << threads::get_thread_description(id);

                // forcefully abort thread, do not throw
                error_code ec(lightweight);
                threads::set_thread_state(id, threads::pending,
                    threads::wait_abort, threads::thread_priority_default, ec);
                if (ec)
                {
                    LERR_(fatal)
                        << "mutex::abort_all:"
                        << " could not abort thread: "
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id << "): " << threads::get_thread_description(id);
                }
            }
        }

    public:
        typedef boost::unique_lock<mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<mutex> scoped_try_lock;

    public:
        mutex()
          : owner_id_(threads::invalid_thread_id_repr)
        {}

        ~mutex()
        {
            if (!queue_.empty())
            {
                LERR_(fatal) << "~mutex: queue is not empty, aborting threads";

                abort_all();
            }
        }

        void lock(char const* description, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return;

            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_repr_type self_id = threads::get_self_id().get();
            if (owner_id_ == threads::invalid_thread_id_repr)
            {
                owner_id_ = self_id;
            } else if(owner_id_ == self_id) {
                HPX_THROWS_IF(ec, deadlock,
                    "mutex::unlock",
                    "The calling thread already owns the mutex");
                return;
            } else {
                // enqueue the request and block this thread
                queue_entry f(self_id);
                queue_.push_back(f);

                reset_queue_entry r(f, queue_);
                {
                    // yield this thread
                    util::scoped_unlock<mutex_type::scoped_lock> unlock(l);
                    this_thread::suspend(threads::suspended, description, ec);
                    if (ec) return;

                    HPX_ASSERT(owner_id_ == f.id_);
                }
            }
        }

        void lock(error_code& ec = throws)
        {
            return lock("mutex::lock", ec);
        }

        bool try_lock(char const* description, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return false;

            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_repr_type self_id = threads::get_self_id().get();
            if (owner_id_ != threads::invalid_thread_id_repr)
                return false;

            owner_id_ = self_id;
            return true;
        }

        bool try_lock(error_code& ec = throws)
        {
            return try_lock("mutex::try_lock", ec);
        }

        bool try_lock_until(boost::posix_time::ptime const& abs_time,
            char const* description, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return false;

            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_repr_type self_id = threads::get_self_id().get();
            if (owner_id_ != threads::invalid_thread_id_repr)
            {
                // enqueue the request and block this thread
                queue_entry f(self_id);
                queue_.push_back(f);

                reset_queue_entry r(f, queue_);
                {
                    // yield this thread
                    util::scoped_unlock<mutex_type::scoped_lock> unlock(l);
                    threads::thread_state_ex_enum const reason =
                        this_thread::suspend(abs_time, description, ec);
                    if (ec) return false;

                    // if the timer has hit, the waiting period timed out
                    if (reason == threads::wait_signaled) //-V110
                        return false;

                    HPX_ASSERT(owner_id_ == f.id_);
                    return true;
                }
            }

            owner_id_ = self_id;
            return true;
        }

        template <typename Clock, typename Duration>
        bool try_lock_until(
            boost::chrono::time_point<Clock, Duration> const& abs_time,
            char const* description, error_code& ec = throws)
        {
            return try_lock_until(util::to_ptime(abs_time), description, ec);
        }

        bool try_lock_until(
            boost::posix_time::ptime const& abs_time,
            error_code& ec = throws)
        {
            return try_lock_until(abs_time, "mutex::try_lock_until", ec);
        }

        template <typename Clock, typename Duration>
        bool try_lock_until(
            boost::chrono::time_point<Clock, Duration> const& abs_time,
            error_code& ec = throws)
        {
            return try_lock_until(util::to_ptime(abs_time), ec);
        }

        bool try_lock_for(
            boost::posix_time::time_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return false;

            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_repr_type self_id = threads::get_self_id().get();
            if (owner_id_ != threads::invalid_thread_id_repr)
            {
                // enqueue the request and block this thread
                queue_entry f(self_id);
                queue_.push_back(f);

                reset_queue_entry r(f, queue_);
                {
                    // yield this thread
                    util::scoped_unlock<mutex_type::scoped_lock> unlock(l);
                    threads::thread_state_ex_enum const reason =
                        this_thread::suspend(rel_time, description, ec);
                    if (ec) return false;

                    // if the timer has hit, the waiting period timed out
                    if (reason == threads::wait_signaled) //-V110
                        return false;

                    HPX_ASSERT(owner_id_ == f.id_);
                    return true;
                }
            }

            owner_id_ = self_id;
            return true;
        }

        template <typename Rep, typename Period>
        bool try_lock_for(
            boost::chrono::duration<Rep, Period> const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return try_lock_for(util::to_time_duration(rel_time),
                description, ec);
        }

        bool try_lock_for(
            boost::posix_time::time_duration const& rel_time,
            error_code& ec = throws)
        {
            return try_lock_for(rel_time, "mutex::try_lock_for", ec);
        }

        template <typename Rep, typename Period>
        bool try_lock_for(
            boost::chrono::duration<Rep, Period> const& rel_time,
            error_code& ec = throws)
        {
            return try_lock_for(util::to_time_duration(rel_time), ec);
        }

        void unlock(error_code& ec = throws)
        {
            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return;

            mutex_type::scoped_lock l(mtx_);

            threads::thread_id_repr_type self_id = threads::get_self_id().get();
            if (HPX_UNLIKELY(owner_id_ != self_id))
            {
                HPX_THROWS_IF(ec, lock_error,
                    "mutex::unlock",
                    "The calling thread does not own the mutex");
                return;
            }

            if (!queue_.empty())
            {
                owner_id_ = queue_.front().id_;
                if (HPX_UNLIKELY(!owner_id_))
                {
                    HPX_THROWS_IF(ec, null_thread_id,
                        "mutex::unlock",
                        "NULL thread id encountered");
                    return;
                }
                queue_.front().id_ = threads::invalid_thread_id_repr;
                queue_.pop_front();

                util::scoped_unlock<mutex_type::scoped_lock> unlock(l);
                threads::set_thread_state(threads::thread_id_type(
                    reinterpret_cast<threads::thread_data_base*>(owner_id_)),
                    threads::pending, threads::wait_timeout,
                    threads::thread_priority_default, ec);
                if (!ec) return;
            } else {
                owner_id_ = threads::invalid_thread_id_repr;
            }
        }

    private:
        mutable mutex_type mtx_;
        threads::thread_id_repr_type owner_id_;
        queue_type queue_;
    };

    typedef mutex timed_mutex;

}}}

#endif
