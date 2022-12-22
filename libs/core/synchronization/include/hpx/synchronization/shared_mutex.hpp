//  (C) Copyright 2006-2008 Anthony Williams
//  (C) Copyright      2011 Bryce Lelbach
//  (C) Copyright 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file shared_mutex.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <mutex>

namespace hpx {

    namespace detail {

        template <typename Mutex = hpx::mutex>
        class shared_mutex
        {
        private:
            using mutex_type = Mutex;

            struct state_data
            {
                unsigned shared_count;
                bool exclusive;
                bool upgrade;
                bool exclusive_waiting_blocked;
            };

            state_data state;
            mutex_type state_change;
            hpx::condition_variable shared_cond;
            hpx::condition_variable exclusive_cond;
            hpx::condition_variable upgrade_cond;

            void release_waiters()
            {
                exclusive_cond.notify_one();
                shared_cond.notify_all();
            }

        public:
            shared_mutex()
              : state{0u, false, false, false}
              , shared_cond()
              , exclusive_cond()
              , upgrade_cond()
            {
            }

            void lock_shared()
            {
                std::unique_lock<mutex_type> lk(state_change);

                while (state.exclusive || state.exclusive_waiting_blocked)
                {
                    shared_cond.wait(lk);
                }

                ++state.shared_count;
            }

            bool try_lock_shared()
            {
                std::unique_lock<mutex_type> lk(state_change);

                if (state.exclusive || state.exclusive_waiting_blocked)
                    return false;

                else
                {
                    ++state.shared_count;
                    return true;
                }
            }

            void unlock_shared()
            {
                std::unique_lock<mutex_type> lk(state_change);

                bool const last_reader = !--state.shared_count;

                if (last_reader)
                {
                    if (state.upgrade)
                    {
                        state.upgrade = false;
                        state.exclusive = true;

                        upgrade_cond.notify_one();
                    }
                    else
                    {
                        state.exclusive_waiting_blocked = false;
                    }

                    release_waiters();
                }
            }

            void lock()
            {
                std::unique_lock<mutex_type> lk(state_change);

                while (state.shared_count || state.exclusive)
                {
                    state.exclusive_waiting_blocked = true;
                    exclusive_cond.wait(lk);
                }

                state.exclusive = true;
            }

            bool try_lock()
            {
                std::unique_lock<mutex_type> lk(state_change);

                if (state.shared_count || state.exclusive)
                    return false;

                else
                {
                    state.exclusive = true;
                    return true;
                }
            }

            void unlock()
            {
                std::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            void lock_upgrade()
            {
                std::unique_lock<mutex_type> lk(state_change);

                while (state.exclusive || state.exclusive_waiting_blocked ||
                    state.upgrade)
                {
                    shared_cond.wait(lk);
                }

                ++state.shared_count;
                state.upgrade = true;
            }

            bool try_lock_upgrade()
            {
                std::unique_lock<mutex_type> lk(state_change);

                if (state.exclusive || state.exclusive_waiting_blocked ||
                    state.upgrade)
                    return false;

                else
                {
                    ++state.shared_count;
                    state.upgrade = true;
                    return true;
                }
            }

            void unlock_upgrade()
            {
                std::unique_lock<mutex_type> lk(state_change);
                state.upgrade = false;
                bool const last_reader = !--state.shared_count;

                if (last_reader)
                {
                    state.exclusive_waiting_blocked = false;
                    release_waiters();
                }
            }

            void unlock_upgrade_and_lock()
            {
                std::unique_lock<mutex_type> lk(state_change);
                --state.shared_count;

                while (state.shared_count)
                {
                    upgrade_cond.wait(lk);
                }

                state.upgrade = false;
                state.exclusive = true;
            }

            void unlock_and_lock_upgrade()
            {
                std::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                state.upgrade = true;
                ++state.shared_count;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            void unlock_and_lock_shared()
            {
                std::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                ++state.shared_count;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            bool try_unlock_shared_and_lock()
            {
                std::unique_lock<mutex_type> lk(state_change);
                if (!state.exclusive && !state.exclusive_waiting_blocked &&
                    !state.upgrade && state.shared_count == 1)
                {
                    state.shared_count = 0;
                    state.exclusive = true;
                    return true;
                }
                return false;
            }

            void unlock_upgrade_and_lock_shared()
            {
                std::unique_lock<mutex_type> lk(state_change);
                state.upgrade = false;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }
        };
    }    // namespace detail

    /// The \a shared_mutex class is a synchronization primitive that can be
    /// used to protect shared data from being simultaneously accessed by
    /// multiple threads. In contrast to other mutex types which facilitate
    /// exclusive access, a \a shared_mutex has two levels of access:
    ///   - \a shared - several threads can share ownership of the same
    ///            mutex.
    ///   - \a exclusive - only one thread can own the mutex.
    ///
    /// \details If one thread has acquired the exclusive lock (through \a lock,
    ///          \a try_lock), no other threads can acquire the lock (including
    ///          the shared). If one thread has acquired the shared lock
    ///          (through \a lock_shared, \a try_lock_shared), no other thread
    ///          can acquire the exclusive lock, but can acquire the shared
    ///          lock. Only when the exclusive lock has not been acquired by any
    ///          thread, the shared lock can be acquired by multiple threads.
    ///          Within one thread, only one lock (shared or exclusive) can be
    ///          acquired at the same time. Shared mutexes are especially useful
    ///          when shared data can be safely read by any number of threads
    ///          simultaneously, but a thread may only write the same data when
    ///          no other thread is reading or writing at the same time. The \a
    ///          shared_mutex class satisfies all requirements of \a SharedMutex
    ///          and \a StandardLayoutType.
    using shared_mutex = detail::shared_mutex<>;
}    // namespace hpx

namespace hpx::lcos::local {

    using shared_mutex HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::shared_mutex is deprecated, use hpx::shared_mutex "
        "instead") = hpx::shared_mutex;
}
