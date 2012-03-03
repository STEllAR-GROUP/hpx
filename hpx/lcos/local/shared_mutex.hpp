//  (C) Copyright 2006-2008 Anthony Williams
//  (C) Copyright      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F)
#define HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/util/unlock_lock.hpp>

namespace hpx { namespace lcos { namespace local
{
    class shared_mutex
    {
    private:
        struct state_data
        {
            unsigned shared_count;
            bool exclusive;
            bool upgrade;
            bool exclusive_waiting_blocked;
        };

        state_data state;
        lcos::local::mutex state_change;
        lcos::local::counting_semaphore shared_cond;
        lcos::local::counting_semaphore exclusive_cond;
        lcos::local::counting_semaphore upgrade_cond;

        void release_waiters()
        {
            exclusive_cond.signal(1);
            shared_cond.signal_all();
        }

      public:
        shared_mutex()
        {
            state_data state_ = {0, 0, 0, 0};
            state = state_;
        }

        void lock_shared()
        {
            lcos::local::mutex::scoped_lock lk(state_change);

            while (state.exclusive || state.exclusive_waiting_blocked)
            {
                util::unlock_the_lock<lcos::local::mutex::scoped_lock> ul(lk);
                shared_cond.wait();
            }

            ++state.shared_count;
        }

        bool try_lock_shared()
        {
            lcos::local::mutex::scoped_lock lk(state_change);

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
            lcos::local::mutex::scoped_lock lk(state_change);

            bool const last_reader = !--state.shared_count;

            if (last_reader)
            {
                if (state.upgrade)
                {
                    state.upgrade = false;
                    state.exclusive = true;
                    upgrade_cond.signal(1);
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
            lcos::local::mutex::scoped_lock lk(state_change);

            while (state.shared_count || state.exclusive)
            {
                state.exclusive_waiting_blocked = true;
                util::unlock_the_lock<lcos::local::mutex::scoped_lock> ul(lk);
                exclusive_cond.wait();
            }

            state.exclusive = true;
        }

        bool try_lock()
        {
            lcos::local::mutex::scoped_lock lk(state_change);

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
            lcos::local::mutex::scoped_lock lk(state_change);
            state.exclusive = false;
            state.exclusive_waiting_blocked = false;
            release_waiters();
        }

        void lock_upgrade()
        {
            lcos::local::mutex::scoped_lock lk(state_change);

            while (state.exclusive || state.exclusive_waiting_blocked || state.upgrade)
            {
                util::unlock_the_lock<lcos::local::mutex::scoped_lock> ul(lk);
                shared_cond.wait();
            }

            ++state.shared_count;
            state.upgrade = true;
        }

        bool try_lock_upgrade()
        {
            lcos::local::mutex::scoped_lock lk(state_change);

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
            lcos::local::mutex::scoped_lock lk(state_change);
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
            lcos::local::mutex::scoped_lock lk(state_change);
            --state.shared_count;

            while (state.shared_count)
            {
                util::unlock_the_lock<lcos::local::mutex::scoped_lock> ul(lk);
                upgrade_cond.wait();
            }

            state.upgrade = false;
            state.exclusive = true;
        }

        void unlock_and_lock_upgrade()
        {
            lcos::local::mutex::scoped_lock lk(state_change);
            state.exclusive = false;
            state.upgrade = true;
            ++state.shared_count;
            state.exclusive_waiting_blocked = false;
            release_waiters();
        }

        void unlock_and_lock_shared()
        {
            lcos::local::mutex::scoped_lock lk(state_change);
            state.exclusive = false;
            ++state.shared_count;
            state.exclusive_waiting_blocked = false;
            release_waiters();
        }

        void unlock_upgrade_and_lock_shared()
        {
            lcos::local::mutex::scoped_lock lk(state_change);
            state.upgrade = false;
            state.exclusive_waiting_blocked = false;
            release_waiters();
        }
    };
}}}

#endif // HPX_F0757EAC_E2A3_4F80_A1EC_8CC7EB55186F

