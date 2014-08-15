//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONDITION_VARIABLE_DEC_4_2013_0130PM)
#define HPX_LCOS_LOCAL_CONDITION_VARIABLE_DEC_4_2013_0130PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/scoped_unlock.hpp>

#include <boost/chrono/time_point.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>

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
            util::ignore_while_checking<Lock> ignore_lock(&lock);
            mutex_type::scoped_lock l(mtx_);
            util::scoped_unlock<Lock> unlock(lock);
            cond_.wait(l, ec);
            l.unlock();
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
            util::ignore_while_checking<Lock> ignore_lock(&lock);
            mutex_type::scoped_lock l(mtx_);
            util::scoped_unlock<Lock> unlock(lock);

            threads::thread_state_ex_enum const reason =
                cond_.wait_until(l, at, ec);
            l.unlock();
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
            util::ignore_while_checking<Lock> ignore_lock(&lock);
            mutex_type::scoped_lock l(mtx_);
            util::scoped_unlock<Lock> unlock(lock);

            threads::thread_state_ex_enum const reason =
                cond_.wait_for(l, p, ec);
            l.unlock();
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

#endif
