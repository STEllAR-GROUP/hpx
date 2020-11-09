//  Copyright (c) 2001-2003 William E. Kempf
//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/functional/bind.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <vector>

std::chrono::milliseconds const timeout_resolution(100);

template <typename M>
struct test_lock
{
    typedef M mutex_type;
    typedef std::unique_lock<M> lock_type;

    void operator()()
    {
        mutex_type mutex;
        hpx::lcos::local::condition_variable_any condition;

        // Test the lock's constructors.
        {
            lock_type lock(mutex, std::defer_lock);
            HPX_TEST(!lock);
        }
        lock_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        std::chrono::system_clock::time_point xt =
            std::chrono::system_clock::now() + std::chrono::milliseconds(10);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(condition.wait_until(lock, xt) ==
            hpx::lcos::local::cv_status::timeout);
        HPX_TEST(lock ? true : false);

        // Test the lock and unlock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
    }
};

template <typename M>
struct test_trylock
{
    typedef M mutex_type;
    typedef std::unique_lock<M> try_lock_type;

    void operator()()
    {
        mutex_type mutex;
        hpx::lcos::local::condition_variable_any condition;

        // Test the lock's constructors.
        {
            try_lock_type lock(mutex);
            HPX_TEST(lock ? true : false);
        }
        {
            try_lock_type lock(mutex, std::defer_lock);
            HPX_TEST(!lock);
        }
        try_lock_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        std::chrono::system_clock::time_point xt =
            std::chrono::system_clock::now() + std::chrono::milliseconds(10);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(condition.wait_until(lock, xt) ==
            hpx::lcos::local::cv_status::timeout);
        HPX_TEST(lock ? true : false);

        // Test the lock, unlock and trylock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);
        HPX_TEST(lock.try_lock());
        HPX_TEST(lock ? true : false);
    }
};

template <typename Mutex>
struct test_lock_times_out_if_other_thread_has_lock
{
    typedef std::unique_lock<Mutex> Lock;

    Mutex m;
    hpx::lcos::local::mutex done_mutex;
    bool done;
    bool locked;
    hpx::lcos::local::condition_variable_any done_cond;

    test_lock_times_out_if_other_thread_has_lock()
      : done(false)
      , locked(false)
    {
    }

    void locking_thread()
    {
        Lock lock(m, std::defer_lock);
        lock.try_lock_for(std::chrono::milliseconds(50));

        std::lock_guard<hpx::lcos::local::mutex> lk(done_mutex);
        locked = lock.owns_lock();
        done = true;
        done_cond.notify_one();
    }

    void locking_thread_through_constructor()
    {
        Lock lock(m, std::chrono::milliseconds(50));

        std::lock_guard<hpx::lcos::local::mutex> lk(done_mutex);
        locked = lock.owns_lock();
        done = true;
        done_cond.notify_one();
    }

    bool is_done() const
    {
        return done;
    }

    typedef test_lock_times_out_if_other_thread_has_lock<Mutex> this_type;

    void do_test(void (this_type::*test_func)())
    {
        Lock lock(m);

        locked = false;
        done = false;

        hpx::thread t(test_func, this);

        try
        {
            {
                std::unique_lock<hpx::lcos::local::mutex> lk(done_mutex);
                HPX_TEST(done_cond.wait_for(lk, std::chrono::seconds(2),
                    hpx::util::bind(&this_type::is_done, this)));
                HPX_TEST(!locked);
            }

            lock.unlock();
            t.join();
        }
        catch (...)
        {
            lock.unlock();
            t.join();
            throw;
        }
    }

    void operator()()
    {
        do_test(&this_type::locking_thread);
        do_test(&this_type::locking_thread_through_constructor);
    }
};

template <typename M>
struct test_timedlock
{
    typedef M mutex_type;
    typedef std::unique_lock<M> try_lock_for_type;

    static bool fake_predicate()
    {
        return false;
    }

    void operator()()
    {
        test_lock_times_out_if_other_thread_has_lock<mutex_type>()();

        mutex_type mutex;
        hpx::lcos::local::condition_variable_any condition;

        // Test the lock's constructors.
        {
            // Construct and initialize an xtime for a fast time out.
            std::chrono::system_clock::time_point xt =
                std::chrono::system_clock::now() +
                std::chrono::milliseconds(10);

            try_lock_for_type lock(mutex, xt);
            HPX_TEST(lock ? true : false);
        }
        {
            try_lock_for_type lock(mutex, std::defer_lock);
            HPX_TEST(!lock);
        }
        try_lock_for_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        std::chrono::system_clock::time_point timeout =
            std::chrono::system_clock::now() + std::chrono::milliseconds(100);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(!condition.wait_until(lock, timeout, fake_predicate));
        HPX_TEST(lock ? true : false);

        std::chrono::system_clock::time_point const now =
            std::chrono::system_clock::now();
        HPX_TEST(timeout - timeout_resolution < now);

        // Test the lock, unlock and timedlock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);

        std::chrono::system_clock::time_point target =
            std::chrono::system_clock::now() + std::chrono::milliseconds(100);
        HPX_TEST(lock.try_lock_until(target));
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);

        HPX_TEST(mutex.try_lock_for(std::chrono::milliseconds(100)));
        mutex.unlock();

        HPX_TEST(lock.try_lock_for(std::chrono::milliseconds(100)));
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);
    }
};

template <typename M>
struct test_recursive_lock
{
    typedef M mutex_type;
    typedef std::unique_lock<M> lock_type;

    void operator()()
    {
        mutex_type mx;
        lock_type lock1(mx);
        lock_type lock2(mx);
    }
};

void test_mutex()
{
    test_lock<hpx::lcos::local::mutex>()();
    test_trylock<hpx::lcos::local::mutex>()();
}

void test_timed_mutex()
{
    test_lock<hpx::lcos::local::timed_mutex>()();
    test_trylock<hpx::lcos::local::timed_mutex>()();
    test_timedlock<hpx::lcos::local::timed_mutex>()();
}

//void test_recursive_mutex()
//{
//    test_lock<hpx::lcos::local::recursive_mutex>()();
//    test_trylock<hpx::lcos::local::recursive_mutex>()();
//    test_recursive_lock<hpx::lcos::local::recursive_mutex>()();
//}
//
//void test_recursive_timed_mutex()
//{
//    test_lock<hpx::lcos::local::recursive_timed_mutex()();
//    test_trylock<hpx::lcos::local::recursive_timed_mutex()();
//    test_timedlock<hpx::lcos::local::recursive_timed_mutex()();
//    test_recursive_lock<hpx::lcos::local::recursive_timed_mutex()();
//}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

int hpx_main(variables_map&)
{
    {
        test_mutex();
        test_timed_mutex();
        //~ test_recursive_mutex();
        //~ test_recursive_timed_mutex();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
