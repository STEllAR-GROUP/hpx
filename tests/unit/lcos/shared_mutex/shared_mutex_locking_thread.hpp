//  (C) Copyright 2008 Anthony Williams
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_SHARED_MUTEX_LOCKING_THREAD_AUG_03_2015_0927PM)
#define HPX_TEST_SHARED_MUTEX_LOCKING_THREAD_AUG_03_2015_0927PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/condition_variable.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>

#include <boost/thread/locks.hpp>

#include <mutex>

namespace test
{
    template <typename Lock>
    class locking_thread
    {
    private:
        hpx::lcos::local::shared_mutex& rw_mutex;
        unsigned& unblocked_count;
        hpx::lcos::local::condition_variable& unblocked_condition;
        unsigned& simultaneous_running_count;
        unsigned& max_simultaneous_running;
        hpx::lcos::local::mutex& unblocked_count_mutex;
        hpx::lcos::local::mutex& finish_mutex;

    public:
        locking_thread(
                hpx::lcos::local::shared_mutex& rw_mutex_,
                unsigned& unblocked_count_,
                hpx::lcos::local::mutex& unblocked_count_mutex_,
                hpx::lcos::local::condition_variable& unblocked_condition_,
                hpx::lcos::local::mutex& finish_mutex_,
                unsigned& simultaneous_running_count_,
                unsigned& max_simultaneous_running_)
          : rw_mutex(rw_mutex_),
            unblocked_count(unblocked_count_),
            unblocked_condition(unblocked_condition_),
            simultaneous_running_count(simultaneous_running_count_),
            max_simultaneous_running(max_simultaneous_running_),
            unblocked_count_mutex(unblocked_count_mutex_),
            finish_mutex(finish_mutex_)
        {}

        void operator()()
        {
            // acquire lock
            Lock lock(rw_mutex);

            // increment count to show we're unblocked
            {
                std::unique_lock<hpx::lcos::local::mutex> ublock(
                    unblocked_count_mutex);

                ++unblocked_count;
                unblocked_condition.notify_one();
                ++simultaneous_running_count;
                if (simultaneous_running_count > max_simultaneous_running)
                {
                    max_simultaneous_running = simultaneous_running_count;
                }
            }

            // wait to finish
            std::unique_lock<hpx::lcos::local::mutex> finish_lock(finish_mutex);
            {
                std::unique_lock<hpx::lcos::local::mutex> ublock(
                    unblocked_count_mutex);

                --simultaneous_running_count;
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    class simple_writing_thread
    {
    private:
        hpx::lcos::local::shared_mutex& rwm;
        hpx::lcos::local::mutex& finish_mutex;
        hpx::lcos::local::mutex& unblocked_mutex;
        unsigned& unblocked_count;

    public:
        simple_writing_thread(
                hpx::lcos::local::shared_mutex& rwm_,
                hpx::lcos::local::mutex& finish_mutex_,
                hpx::lcos::local::mutex& unblocked_mutex_,
                unsigned& unblocked_count_)
          : rwm(rwm_),
            finish_mutex(finish_mutex_),
            unblocked_mutex(unblocked_mutex_),
            unblocked_count(unblocked_count_)
        {}

        void operator()()
        {
            std::unique_lock<hpx::lcos::local::shared_mutex>  lk(rwm);
            {
                std::unique_lock<hpx::lcos::local::mutex> ulk(unblocked_mutex);
                ++unblocked_count;
            }
            std::unique_lock<hpx::lcos::local::mutex> flk(finish_mutex);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    class simple_reading_thread
    {
    private:
        hpx::lcos::local::shared_mutex& rwm;
        hpx::lcos::local::mutex& finish_mutex;
        hpx::lcos::local::mutex& unblocked_mutex;
        unsigned& unblocked_count;

    public:
        simple_reading_thread(
                hpx::lcos::local::shared_mutex& rwm_,
                hpx::lcos::local::mutex& finish_mutex_,
                hpx::lcos::local::mutex& unblocked_mutex_,
                unsigned& unblocked_count_)
          : rwm(rwm_),
            finish_mutex(finish_mutex_),
            unblocked_mutex(unblocked_mutex_),
            unblocked_count(unblocked_count_)
        {}

        void operator()()
        {
            boost::shared_lock<hpx::lcos::local::shared_mutex>  lk(rwm);
            {
                std::unique_lock<hpx::lcos::local::mutex> ulk(unblocked_mutex);
                ++unblocked_count;
            }
            std::unique_lock<hpx::lcos::local::mutex> flk(finish_mutex);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    class simple_upgrade_thread
    {
    private:
        hpx::lcos::local::shared_mutex& rwm;
        hpx::lcos::local::mutex& finish_mutex;
        hpx::lcos::local::mutex& unblocked_mutex;
        unsigned& unblocked_count;

    public:
        simple_upgrade_thread(
                hpx::lcos::local::shared_mutex& rwm_,
                hpx::lcos::local::mutex& finish_mutex_,
                hpx::lcos::local::mutex& unblocked_mutex_,
                unsigned& unblocked_count_)
          : rwm(rwm_),
            finish_mutex(finish_mutex_),
            unblocked_mutex(unblocked_mutex_),
            unblocked_count(unblocked_count_)
        {}

        void operator()()
        {
            boost::upgrade_lock<hpx::lcos::local::shared_mutex> lk(rwm);
            {
                std::unique_lock<hpx::lcos::local::mutex> ulk(unblocked_mutex);
                ++unblocked_count;
            }
            std::unique_lock<hpx::lcos::local::mutex> flk(finish_mutex);
        }
    };
}

#endif
