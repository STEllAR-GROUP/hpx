//  (C) Copyright 2006-2008 Anthony Williams
//  (C) Copyright      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F)
#define HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/detail/counting_semaphore.hpp>
#include <hpx/lcos/local/no_mutex.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename Mutex = lcos::local::mutex>
        class shared_mutex
        {
        private:
            typedef Mutex mutex_type;

            struct state_data
            {
                unsigned shared_count;
                bool exclusive;
                bool upgrade;
                bool exclusive_waiting_blocked;
            };

            state_data state;
            mutex_type state_change;
            lcos::local::detail::counting_semaphore shared_cond;
            lcos::local::detail::counting_semaphore exclusive_cond;
            lcos::local::detail::counting_semaphore upgrade_cond;

            void release_waiters()
            {
                no_mutex mtx;
                {
                    boost::unique_lock<no_mutex> l(mtx);
                    exclusive_cond.signal(std::move(l), 1);
                }
                {
                    boost::unique_lock<no_mutex> l(mtx);
                    shared_cond.signal_all(std::move(l));
                }
            }

        public:
            shared_mutex()
            {
                state_data state_ = {0, 0, 0, 0};
                state = state_;
            }

            void lock_shared()
            {
                boost::unique_lock<mutex_type> lk(state_change);

                while (state.exclusive || state.exclusive_waiting_blocked)
                {
                    shared_cond.wait(lk, 1);
                }

                ++state.shared_count;
            }

            bool try_lock_shared()
            {
                boost::unique_lock<mutex_type> lk(state_change);

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
                boost::unique_lock<mutex_type> lk(state_change);

                bool const last_reader = !--state.shared_count;

                if (last_reader)
                {
                    if (state.upgrade)
                    {
                        state.upgrade = false;
                        state.exclusive = true;

                        no_mutex mtx;
                        boost::unique_lock<no_mutex> l(mtx);
                        upgrade_cond.signal(std::move(l), 1);
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
                boost::unique_lock<mutex_type> lk(state_change);

                while (state.shared_count || state.exclusive)
                {
                    state.exclusive_waiting_blocked = true;
                    exclusive_cond.wait(lk, 1);
                }

                state.exclusive = true;
            }

            bool try_lock()
            {
                boost::unique_lock<mutex_type> lk(state_change);

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
                boost::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            void lock_upgrade()
            {
                boost::unique_lock<mutex_type> lk(state_change);

                while (state.exclusive || state.exclusive_waiting_blocked
                    || state.upgrade)
                {
                    shared_cond.wait(lk, 1);
                }

                ++state.shared_count;
                state.upgrade = true;
            }

            bool try_lock_upgrade()
            {
                boost::unique_lock<mutex_type> lk(state_change);

                if (state.exclusive || state.exclusive_waiting_blocked || state.upgrade)
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
                boost::unique_lock<mutex_type> lk(state_change);
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
                boost::unique_lock<mutex_type> lk(state_change);
                --state.shared_count;

                while (state.shared_count)
                {
                    upgrade_cond.wait(lk, 1);
                }

                state.upgrade = false;
                state.exclusive = true;
            }

            void unlock_and_lock_upgrade()
            {
                boost::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                state.upgrade = true;
                ++state.shared_count;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            void unlock_and_lock_shared()
            {
                boost::unique_lock<mutex_type> lk(state_change);
                state.exclusive = false;
                ++state.shared_count;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }

            bool try_unlock_shared_and_lock()
            {
                boost::unique_lock<mutex_type> lk(state_change);
                if(    !state.exclusive
                    && !state.exclusive_waiting_blocked
                    && !state.upgrade
                    && state.shared_count == 1)
                {
                    state.shared_count=0;
                    state.exclusive = true;
                    return true;
                }
                return false;
            }

            void unlock_upgrade_and_lock_shared()
            {
                boost::unique_lock<mutex_type> lk(state_change);
                state.upgrade = false;
                state.exclusive_waiting_blocked = false;
                release_waiters();
            }
        };
    }

    typedef detail::shared_mutex<> shared_mutex;
}}}

#endif // HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F

