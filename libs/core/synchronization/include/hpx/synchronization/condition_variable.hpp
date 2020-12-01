//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/detail/condition_variable.hpp>
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
        using mutex_type = detail::condition_variable_data::mutex_type;
        using data_type =
            hpx::memory::intrusive_ptr<detail::condition_variable_data>;

    public:
        condition_variable()
          : data_(data_type(new detail::condition_variable_data, false))
        {
        }

        // Preconditions: There is no thread blocked on *this. [Note: That is,
        //      all threads have been notified; they could subsequently block
        //      on the lock specified in the wait.This relaxes the usual rules,
        //      which would have required all wait calls to happen before
        //      destruction.Only the notification to unblock the wait needs to
        //      happen before destruction.The user should take care to ensure
        //      that no threads wait on *this once the destructor has been
        //      started, especially when the waiting threads are calling the
        //      wait functions in a loop or using the overloads of wait,
        //      wait_for, or wait_until that take a predicate. end note]
        //
        // IOW, ~condition_variable() can execute before a signaled thread
        // returns from a wait. If this happens with condition_variable, that
        // waiting thread will attempt to lock the destructed mutex.
        // To fix this, there must be shared ownership of the data members
        // between the condition_variable_any object and the member functions
        // wait (wait_for, etc.).
        ~condition_variable() = default;

        void notify_one(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(data_->mtx_);
            data_->cond_.notify_one(std::move(l), ec);
        }

        void notify_all(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(data_->mtx_);
            data_->cond_.notify_all(std::move(l), ec);
        }

        void wait(std::unique_lock<mutex>& lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            auto data = data_;    // keep data alive

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(data->mtx_);
            util::unlock_guard<std::unique_lock<mutex>> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            data->cond_.wait(l, ec);
        }

        template <typename Predicate>
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
            hpx::chrono::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            auto data = data_;    // keep data alive

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(data->mtx_);
            util::unlock_guard<std::unique_lock<mutex>> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            threads::thread_restart_state const reason =
                data->cond_.wait_until(l, abs_time, ec);

            if (ec)
                return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason ==
                       threads::thread_restart_state::timeout) ?    //-V110
                cv_status::timeout :
                cv_status::no_timeout;
        }

        template <typename Predicate>
        bool wait_until(std::unique_lock<mutex>& lock,
            hpx::chrono::steady_time_point const& abs_time, Predicate pred,
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
            hpx::chrono::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), ec);
        }

        template <typename Predicate>
        bool wait_for(std::unique_lock<mutex>& lock,
            hpx::chrono::steady_duration const& rel_time, Predicate pred,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), std::move(pred), ec);
        }

    private:
        hpx::util::cache_aligned_data_derived<data_type> data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class condition_variable_any
    {
    private:
        using mutex_type = detail::condition_variable_data::mutex_type;
        using data_type =
            hpx::memory::intrusive_ptr<detail::condition_variable_data>;

    public:
        condition_variable_any()
          : data_(data_type(new detail::condition_variable_data, false))
        {
        }

        // Preconditions: There is no thread blocked on *this. [Note: That is,
        //      all threads have been notified; they could subsequently block
        //      on the lock specified in the wait.This relaxes the usual rules,
        //      which would have required all wait calls to happen before
        //      destruction.Only the notification to unblock the wait needs to
        //      happen before destruction.The user should take care to ensure
        //      that no threads wait on *this once the destructor has been
        //      started, especially when the waiting threads are calling the
        //      wait functions in a loop or using the overloads of wait,
        //      wait_for, or wait_until that take a predicate. end note]
        //
        // IOW, ~condition_variable_any() can execute before a signaled thread
        // returns from a wait. If this happens with condition_variable, that
        // waiting thread will attempt to lock the destructed mutex.
        // To fix this, there must be shared ownership of the data members
        // between the condition_variable_any object and the member functions
        // wait (wait_for, etc.).
        ~condition_variable_any() = default;

        void notify_one(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(data_->mtx_);
            data_->cond_.notify_one(std::move(l), ec);
        }

        void notify_all(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(data_->mtx_);
            data_->cond_.notify_all(std::move(l), ec);
        }

        template <typename Lock>
        void wait(Lock& lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            auto data = data_;    // keep data alive

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(data->mtx_);
            util::unlock_guard<Lock> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            data->cond_.wait(l, ec);
        }

        template <typename Lock, typename Predicate>
        void wait(Lock& lock, Predicate pred, error_code& /* ec */ = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            while (!pred())
            {
                wait(lock);
            }
        }

        template <typename Lock>
        cv_status wait_until(Lock& lock,
            hpx::chrono::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            auto data = data_;    // keep data alive

            util::ignore_all_while_checking ignore_lock;
            std::unique_lock<mutex_type> l(data->mtx_);
            util::unlock_guard<Lock> unlock(lock);

            // The following ensures that the inner lock will be unlocked
            // before the outer to avoid deadlock (fixes issue #3608)
            std::lock_guard<std::unique_lock<mutex_type>> unlock_next(
                l, std::adopt_lock);

            threads::thread_restart_state const reason =
                data->cond_.wait_until(l, abs_time, ec);

            if (ec)
                return cv_status::error;

            // if the timer has hit, the waiting period timed out
            return (reason ==
                       threads::thread_restart_state::timeout) ?    //-V110
                cv_status::timeout :
                cv_status::no_timeout;
        }

        template <typename Lock, typename Predicate>
        bool wait_until(Lock& lock,
            hpx::chrono::steady_time_point const& abs_time, Predicate pred,
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

        template <typename Lock>
        cv_status wait_for(Lock& lock,
            hpx::chrono::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), ec);
        }

        template <typename Lock, typename Predicate>
        bool wait_for(Lock& lock, hpx::chrono::steady_duration const& rel_time,
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

            auto data = data_;    // keep data alive

            auto f = [&data, &ec] {
                std::unique_lock<mutex_type> l(data->mtx_);
                data->cond_.notify_all(std::move(l), ec);
            };
            stop_callback<decltype(f)> cb(stoken, std::move(f));

            while (!pred())
            {
                util::ignore_all_while_checking ignore_lock;
                std::unique_lock<mutex_type> l(data->mtx_);
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

                data->cond_.wait(l, ec);
            }

            return true;
        }

        template <typename Lock, typename Predicate>
        bool wait_until(Lock& lock, stop_token stoken,
            hpx::chrono::steady_time_point const& abs_time, Predicate pred,
            error_code& ec = throws)
        {
            if (stoken.stop_requested())
            {
                return pred();
            }

            auto data = data_;    // keep data alive

            auto f = [&data, &ec] {
                std::unique_lock<mutex_type> l(data->mtx_);
                data->cond_.notify_all(std::move(l), ec);
            };
            stop_callback<decltype(f)> cb(stoken, std::move(f));

            while (!pred())
            {
                bool should_stop;
                {
                    util::ignore_all_while_checking ignore_lock;
                    std::unique_lock<mutex_type> l(data->mtx_);
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

                    threads::thread_restart_state const reason =
                        data->cond_.wait_until(l, abs_time, ec);

                    if (ec)
                        return false;

                    should_stop =
                        (reason == threads::thread_restart_state::timeout) ||
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
            hpx::chrono::steady_duration const& rel_time, Predicate pred,
            error_code& ec = throws)
        {
            return wait_until(
                lock, stoken, rel_time.from_now(), std::move(pred), ec);
        }

    private:
        hpx::util::cache_aligned_data_derived<data_type> data_;
    };
}}}    // namespace hpx::lcos::local
