//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONDITION_VARIABLE_DEC_4_2013_0130PM)
#define HPX_LCOS_LOCAL_CONDITION_VARIABLE_DEC_4_2013_0130PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/noncopyable.hpp>
#include <boost/utility/declval.hpp>

#define HPX_ASSERT_OWNS_LOCK(l) ::hpx::lcos::local::detail::assert_owns_lock(l, 0L)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    template <typename Lock>
    void assert_owns_lock(Lock& l, int)
    {}

#   if !defined(BOOST_NO_CXX11_DECLTYPE_N3276) && !defined(BOOST_NO_SFINAE_EXPR)
    template <typename Lock>
    decltype(boost::declval(Lock).owns_lock())
    assert_owns_lock(Lock& l, long)
    {
        HPX_ASSERT(l.owns_lock());
        return true;
    }
#   else
    template <typename Mutex>
    void assert_owns_lock(boost::unique_lock<Mutex>& l, long)
    {
        HPX_ASSERT(l.owns_lock());
    }
#   endif

    ///////////////////////////////////////////////////////////////////////////
    class condition_variable : boost::noncopyable
    {
    private:
        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(threads::thread_id_type const& id)
              : id_(id)
            {}

            threads::thread_id_type id_;
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

    public:
        condition_variable()
        {}

        ~condition_variable()
        {
            if (!queue_.empty())
            {
                LERR_(fatal) << "~condition_variable: queue is not empty, aborting threads";

                local::no_mutex no_mtx;
                abort_all(no_mtx);
            }
        }

        template <typename Lock>
        bool empty(Lock& lock) const
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            return queue_.empty();
        }

        template <typename Lock>
        std::size_t size(Lock& lock) const
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            return queue_.size();
        }

        template <typename Lock>
        void notify_one(Lock& lock, error_code& ec = throws) // leaves the lock unlocked
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            if (!queue_.empty())
            {
                threads::thread_id_type id = queue_.front().id_;
                if (HPX_UNLIKELY(!id))
                {
                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_one",
                        "NULL thread id encountered");
                    return;
                }
                queue_.front().id_ = threads::invalid_thread_id;
                queue_.pop_front();

                lock.unlock();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return;
            }
        }

        template <typename Lock>
        void notify_all(Lock& lock, error_code& ec = throws) // leaves the lock unlocked
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            // swap the list
            queue_type queue;
            queue.swap(queue_);
            lock.unlock();

            while (!queue.empty())
            {
                threads::thread_id_type id = queue.front().id_;
                if (HPX_UNLIKELY(!id))
                {
                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_all",
                        "NULL thread id encountered");
                    return;
                }
                queue.front().id_ = threads::invalid_thread_id;
                queue.pop_front();

                threads::set_thread_state(id, threads::pending,
                    threads::wait_timeout, threads::thread_priority_default, ec);
                if (ec) return;
            }
        }

        template <typename Lock>
        void abort_all(Lock& lock) // leaves the lock unlocked
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            // swap the list
            queue_type queue;
            queue.swap(queue_);
            lock.unlock();

            while (!queue.empty())
            {
                threads::thread_id_type id = queue.front().id_;
                queue.front().id_ = threads::invalid_thread_id;
                queue.pop_front();

                // we know that the id is actually the pointer to the thread
                LERR_(fatal)
                        << "condition_variable::abort_all:"
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
                        << "condition_variable::abort_all:"
                        << " could not abort thread: "
                        << get_thread_state_name(threads::get_thread_state(id))
                        << "(" << id << "): " << threads::get_thread_description(id);
                }
            }
        }

        template <typename Lock>
        void wait(Lock& lock, 
            char const* description, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return;

            // enqueue the request and block this thread
            queue_entry f(threads::get_self_id());
            queue_.push_back(f);

            reset_queue_entry r(f, queue_);
            {
                // yield this thread
                util::scoped_unlock<Lock> unlock(lock);
                this_thread::suspend(threads::suspended, description, ec);
                if (ec) return;
            }
        }
        
        template <typename Lock>
        void wait(Lock& lock, error_code& ec = throws)
        {
            return wait(lock, "condition_variable::wait", ec);
        }

        template <typename Lock>
        threads::thread_state_ex_enum
        wait_for(Lock& lock, boost::posix_time::time_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return threads::wait_unknown;

            // enqueue the request and block this thread
            queue_entry f(threads::get_self_id());
            queue_.push_back(f);

            reset_queue_entry r(f, queue_);
            {
                // yield this thread
                util::scoped_unlock<Lock> unlock(lock);
                threads::thread_state_ex_enum const reason =
                    this_thread::suspend(rel_time, description, ec);
                if (ec) return threads::wait_unknown;

                return reason;
            }
        }

        template <typename Lock>
        threads::thread_state_ex_enum
        wait_for(Lock& lock, boost::posix_time::time_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_for(lock, rel_time, 
                "condition_variable::wait_for", ec);
        }

        template <typename Lock>
        threads::thread_state_ex_enum
        wait_until(Lock& lock, boost::posix_time::ptime const& abs_time,
            char const* description, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            threads::thread_self* self = threads::get_self_ptr_checked(ec);
            if (0 == self || ec) return threads::wait_unknown;

            // enqueue the request and block this thread
            queue_entry f(threads::get_self_id());
            queue_.push_back(f);

            reset_queue_entry r(f, queue_);
            {
                // yield this thread
                util::scoped_unlock<Lock> unlock(lock);
                threads::thread_state_ex_enum const reason =
                    this_thread::suspend(abs_time, description, ec);
                if (ec) return threads::wait_unknown;

                return reason;
            }
        }

        template <typename Lock>
        threads::thread_state_ex_enum
        wait_until(Lock& lock, boost::posix_time::ptime const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(lock, abs_time, 
                "condition_variable::wait_until", ec);
        }

    private:
        queue_type queue_;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    BOOST_SCOPED_ENUM_START(cv_status)
    {
        no_timeout, timeout, error
    };
    BOOST_SCOPED_ENUM_END

    class condition_variable
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        void notify_one(error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);
            cond_.notify_one(l, ec);
        }

        void notify_all(error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);
            cond_.notify_all(l, ec);
        }

        template <class Lock>
        void wait(Lock& lock, error_code& ec = throws)
        {
            util::scoped_unlock<Lock> unlock(lock);
            mutex_type::scoped_lock l(mtx_);
            cond_.wait(l, ec);
        }

        template <class Lock, class Predicate>
        void wait(Lock& lock, Predicate pred, error_code& ec = throws)
        {
            while (!pred())
            {
                wait(lock);
            }
        }

        template <typename Lock>
        BOOST_SCOPED_ENUM(cv_status)
        wait_until(Lock& lock,
            boost::posix_time::ptime const& at, error_code& ec = throws)
        {
            util::scoped_unlock<Lock> unlock(lock);
            mutex_type::scoped_lock l(mtx_);

            threads::thread_state_ex_enum const reason =
                cond_.wait_until(l, at, ec);
            if (ec) return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_signaled) ? //-V110
                cv_status::timeout : cv_status::no_timeout;
        }

        template <typename Lock, typename Clock, typename Duration>
        BOOST_SCOPED_ENUM(cv_status)
        wait_until(Lock& lock,
            boost::chrono::time_point<Clock, Duration> const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(lock, util::to_ptime(abs_time), ec);
        }

        template <typename Lock, typename Predicate>
        bool wait_until(Lock& lock,
            boost::posix_time::ptime const& at,
            Predicate pred, error_code& ec = throws)
        {
            while (!pred())
            {
                if (wait_until(lock, at, ec) == cv_status::timeout)
                    return pred();
            }
            return true;
        }

        template <typename Lock, typename Clock, typename Duration, typename Predicate>
        bool wait_until(Lock& lock,
            boost::chrono::time_point<Clock, Duration> const& abs_time,
            Predicate pred, error_code& ec = throws)
        {
            return wait_until(lock, util::to_ptime(abs_time), pred, ec);
        }

        template <typename Lock>
        BOOST_SCOPED_ENUM(cv_status)
        wait_for(Lock& lock,
            boost::posix_time::time_duration const& p,
            error_code& ec = throws)
        {
            util::scoped_unlock<Lock> unlock(lock);
            mutex_type::scoped_lock l(mtx_);

            threads::thread_state_ex_enum const reason =
                cond_.wait_for(l, p, ec);
            if (ec) return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_signaled) ? //-V110
                cv_status::timeout : cv_status::no_timeout;
        }

        template <typename Lock, typename Rep, typename Period>
        BOOST_SCOPED_ENUM(cv_status)
        wait_for(Lock& lock,
            boost::chrono::duration<Rep, Period> const& rel_time,
            error_code& ec = throws)
        {
            return wait_for(lock, util::to_time_duration(rel_time), ec);
        }

        template <typename Lock, typename Predicate>
        bool wait_for(Lock& lock,
            boost::posix_time::time_duration const& p,
            Predicate pred, error_code& ec = throws)
        {
            boost::posix_time::ptime const deadline =
                boost::posix_time::microsec_clock::local_time() + p;
            while (!pred())
            {
                boost::posix_time::ptime const now =
                    boost::posix_time::microsec_clock::local_time();
                if (wait_for(lock, deadline - now, ec) == cv_status::timeout)
                    return pred();
            }
            return true;
        }

        template <typename Lock, typename Rep, typename Period, typename Predicate>
        bool wait_for(Lock& lock,
            boost::chrono::duration<Rep, Period> const& rel_time,
            Predicate pred, error_code& ec = throws)
        {
            return wait_for(lock, util::to_time_duration(rel_time), pred, ec);
        }

    private:
        mutable mutex_type mtx_;
        detail::condition_variable cond_;
    };
}}}

#undef HPX_ASSERT_OWNS_LOCK

#endif
