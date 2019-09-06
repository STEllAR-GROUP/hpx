//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_CONDITION_VARIABLE_HPP
#define HPX_LCOS_LOCAL_CONDITION_VARIABLE_HPP

#include <hpx/config.hpp>
#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    enum class cv_status
    {
        no_timeout, timeout, error
    };

    class condition_variable
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        void notify_one(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            cond_.data_.notify_one(std::move(l), ec);
        }

        void notify_all(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            cond_.data_.notify_all(std::move(l), ec);
        }

        void wait(std::unique_lock<mutex>& lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);
            util::ignore_while_checking<std::unique_lock<mutex> > il(&lock);
            std::unique_lock<mutex_type> l(mtx_.data_);

            util::ignore_while_checking<std::unique_lock<mutex_type>> iil(&l);
            util::unlock_guard<std::unique_lock<mutex> > unlock(lock);
            //The following ensures that the inner lock will be unlocked
            //before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type> > unlock_next(
                l, std::adopt_lock);

            cond_.data_.wait(l, ec);

            // We need to ignore our internal mutex for the user provided lock
            // being able to be reacquired without a lock held during suspension
            // error. We can't use RAII here since the guard object would get
            // destructed before the unlock_guard.
            hpx::util::ignore_lock(&mtx_.data_);
        }

        template <class Predicate>
        void wait(std::unique_lock<mutex>& lock, Predicate pred,
            error_code& /*ec*/ = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            while (!pred())
            {
                wait(lock);
            }
        }

        cv_status wait_until(std::unique_lock<mutex>& lock,
            util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::ignore_while_checking<std::unique_lock<mutex> > il(&lock);
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> iil(&l);
            util::unlock_guard<std::unique_lock<mutex> > unlock(lock);
            //The following ensures that the inner lock will be unlocked
            //before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type> > unlock_next(
                l, std::adopt_lock);

            threads::thread_state_ex_enum const reason =
                cond_.data_.wait_until(l, abs_time, ec);

            // We need to ignore our internal mutex for the user provided lock
            // being able to be reacquired without a lock held during suspension
            // error. We can't use RAII here since the guard object would get
            // destructed before the unlock_guard.
            hpx::util::ignore_lock(&mtx_.data_);

            if (ec) return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_timeout) ? //-V110
                cv_status::timeout : cv_status::no_timeout;
        }

        template <typename Predicate>
        bool wait_until(std::unique_lock<mutex>& lock,
            util::steady_time_point const& abs_time, Predicate pred,
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            while (!pred())
            {
                if (wait_until(lock, abs_time, ec) == cv_status::timeout)
                    return pred();
            }
            return true;
        }

        cv_status wait_for(std::unique_lock<mutex>& lock,
            util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), ec);
        }

        template <typename Predicate>
        bool wait_for(std::unique_lock<mutex>& lock,
            util::steady_duration const& rel_time, Predicate pred,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), std::move(pred), ec);
        }

    private:
        mutable util::cache_line_data<mutex_type> mtx_;
        util::cache_line_data<detail::condition_variable> cond_;
    };

    class condition_variable_any
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        void notify_one(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            cond_.data_.notify_one(std::move(l), ec);
        }

        void notify_all(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            cond_.data_.notify_all(std::move(l), ec);
        }

        template <class Lock>
        void wait(Lock& lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::unlock_guard<Lock> unlock(lock);
            //The following ensures that the inner lock will be unlocked
            //before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type> > unlock_next(
                l, std::adopt_lock);

            cond_.data_.wait(l, ec);

            // We need to ignore our internal mutex for the user provided lock
            // being able to be reacquired without a lock held during suspension
            // error. We can't use RAII here since the guard object would get
            // destructed before the unlock_guard.
            hpx::util::ignore_lock(&mtx_.data_);
        }

        template <class Lock, class Predicate>
        void wait(Lock& lock, Predicate pred, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            while (!pred())
            {
                wait(lock);
            }
        }

        template <typename Lock>
        cv_status
        wait_until(Lock& lock, util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::unlock_guard<Lock> unlock(lock);
            //The following ensures that the inner lock will be unlocked
            //before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type> > unlock_next(
                l, std::adopt_lock);

            threads::thread_state_ex_enum const reason =
                cond_.data_.wait_until(l, abs_time, ec);

            // We need to ignore our internal mutex for the user provided lock
            // being able to be reacquired without a lock held during suspension
            // error. We can't use RAII here since the guard object would get
            // destructed before the unlock_guard.
            hpx::util::ignore_lock(&mtx_.data_);

            if (ec) return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_timeout) ? //-V110
                cv_status::timeout : cv_status::no_timeout;
        }

        template <typename Lock, typename Predicate>
        bool wait_until(Lock& lock, util::steady_time_point const& abs_time,
            Predicate pred, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            while (!pred())
            {
                if (wait_until(lock, abs_time, ec) == cv_status::timeout)
                    return pred();
            }
            return true;
        }

        template <typename Lock>
        cv_status
        wait_for(Lock& lock, util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), ec);
        }

        template <typename Lock, typename Predicate>
        bool wait_for(Lock& lock, util::steady_duration const& rel_time,
            Predicate pred, error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), std::move(pred), ec);
        }

    private:
        mutable util::cache_line_data<mutex_type> mtx_;
        util::cache_line_data<detail::condition_variable> cond_;
    };
}}}

#endif /*HPX_LCOS_LOCAL_CONDITION_VARIABLE_HPP*/
