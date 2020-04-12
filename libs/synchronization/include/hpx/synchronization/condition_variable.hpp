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
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/thread_support/assert_owns_lock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    enum class cv_status
    {
        no_timeout,
        timeout,
        error
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

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::unlock_guard<std::unique_lock<mutex>> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            cond_.data_.wait(l, ec);
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
            util::steady_time_point const& abs_time, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::unlock_guard<std::unique_lock<mutex>> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            threads::thread_state_ex_enum const reason =
                cond_.data_.wait_until(l, abs_time, ec);

            if (ec)
                return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_timeout) ?    //-V110
                cv_status::timeout :
                cv_status::no_timeout;
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
            util::steady_duration const& rel_time, error_code& ec = throws)
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

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            cond_.data_.wait(l, ec);
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
        cv_status wait_until(Lock& lock,
            util::steady_time_point const& abs_time, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(mtx_.data_);
            util::unlock_guard<Lock> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            threads::thread_state_ex_enum const reason =
                cond_.data_.wait_until(l, abs_time, ec);

            if (ec)
                return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason == threads::wait_timeout) ?    //-V110
                cv_status::timeout :
                cv_status::no_timeout;
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
        cv_status wait_for(Lock& lock, util::steady_duration const& rel_time,
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

        // 32.6.4.2, interruptible waits
        template <typename Lock, typename Predicate>
        bool wait(Lock& lock, stop_token stoken, Predicate pred,
            error_code& ec = throws)
        {
            if (stoken.stop_requested())
            {
                return pred();
            }

            auto f = [&] {
                std::unique_lock<mutex_type> l(mtx_.data_);
                cond_.data_.notify_all(std::move(l), ec);
            };
            stop_callback<decltype(f)> cb(stoken, std::move(f));

            while (!pred())
            {
                util::ignore_all_while_checking ignore_lock;
                std::unique_lock<mutex_type> l(mtx_.data_);
                if (stoken.stop_requested())
                {
                    // pred() has already evaluated to false since we last
                    // a acquired lock
                    return false;
                }

                util::unlock_guard<Lock> unlock(lock);

                // The following ensures that the inner lock will be unlocked
                // before the outer to avoid deadlock (fixes issue #3608)
                std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                    l, std::adopt_lock);

                cond_.data_.wait(l, ec);
            }

            return true;
        }

        template <typename Lock, typename Predicate>
        bool wait_until(Lock& lock, stop_token stoken,
            util::steady_time_point const& abs_time, Predicate pred,
            error_code& ec = throws)
        {
            if (stoken.stop_requested())
            {
                return pred();
            }

            auto f = [&] {
                std::unique_lock<mutex_type> l(mtx_.data_);
                cond_.data_.notify_all(std::move(l), ec);
            };
            stop_callback<decltype(f)> cb(stoken, std::move(f));

            while (!pred())
            {
                bool should_stop;
                {
                    util::ignore_all_while_checking ignore_lock;
                    std::unique_lock<mutex_type> l(mtx_.data_);
                    if (stoken.stop_requested())
                    {
                        // pred() has already evaluated to false since we last
                        // acquired lock.
                        return false;
                    }

                    util::unlock_guard<Lock> unlock(lock);

                    // The following ensures that the inner lock will be unlocked
                    // before the outer to avoid deadlock (fixes issue #3608)
                    std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                        l, std::adopt_lock);

                    threads::thread_state_ex_enum const reason =
                        cond_.data_.wait_until(l, abs_time, ec);

                    if (ec)
                        return false;

                    should_stop = (reason == threads::wait_timeout) ||
                        stoken.stop_requested();
                }

                if (should_stop)
                {
                    return pred();
                }
            }
            return true;
        }

        template <typename Lock, typename Predicate>
        bool wait_for(Lock& lock, stop_token stoken,
            util::steady_duration const& rel_time, Predicate pred,
            error_code& ec = throws)
        {
            return wait_until(
                lock, stoken, rel_time.from_now(), std::move(pred), ec);
        }

    private:
        mutable util::cache_line_data<mutex_type> mtx_;
        util::cache_line_data<detail::condition_variable> cond_;
    };
}}}    // namespace hpx::lcos::local

#endif /*HPX_LCOS_LOCAL_CONDITION_VARIABLE_HPP*/
