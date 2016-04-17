//  Copyright (c) 2001-2003 William E. Kempf
//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/local/condition_variable.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/locks.hpp>

#include <string>
#include <vector>

boost::chrono::milliseconds const delay(1000);
boost::chrono::milliseconds const timeout_resolution(100);

template <typename M>
struct test_lock
{
    typedef M mutex_type;
    typedef boost::unique_lock<M> lock_type;

    void operator()()
    {
        mutex_type mutex;
        hpx::lcos::local::condition_variable_any condition;

        // Test the lock's constructors.
        {
            lock_type lock(mutex, boost::defer_lock);
            HPX_TEST(!lock);
        }
        lock_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        boost::chrono::system_clock::time_point xt =
            boost::chrono::system_clock::now()
          + boost::chrono::milliseconds(10);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(condition.wait_until(lock, xt) == hpx::lcos::local::cv_status::timeout);
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
    typedef boost::unique_lock<M> try_lock_type;

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
            try_lock_type lock(mutex, boost::defer_lock);
            HPX_TEST(!lock);
        }
        try_lock_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        boost::chrono::system_clock::time_point xt =
            boost::chrono::system_clock::now()
          + boost::chrono::milliseconds(10);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(condition.wait_until(lock, xt) == hpx::lcos::local::cv_status::timeout);
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

template<typename Mutex>
struct test_lock_times_out_if_other_thread_has_lock
{
    typedef boost::unique_lock<Mutex> Lock;

    Mutex m;
    hpx::lcos::local::mutex done_mutex;
    bool done;
    bool locked;
    hpx::lcos::local::condition_variable_any done_cond;

    test_lock_times_out_if_other_thread_has_lock():
        done(false),locked(false)
    {}

    void locking_thread()
    {
        Lock lock(m,boost::defer_lock);
        lock.try_lock_for(boost::chrono::milliseconds(50));

        boost::lock_guard<hpx::lcos::local::mutex> lk(done_mutex);
        locked=lock.owns_lock();
        done=true;
        done_cond.notify_one();
    }

    void locking_thread_through_constructor()
    {
        Lock lock(m,boost::chrono::milliseconds(50));

        boost::lock_guard<hpx::lcos::local::mutex> lk(done_mutex);
        locked=lock.owns_lock();
        done=true;
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

        locked=false;
        done=false;

        hpx::thread t(test_func,this);

        try
        {
            {
                boost::unique_lock<hpx::lcos::local::mutex> lk(done_mutex);
                HPX_TEST(done_cond.wait_for(lk,boost::chrono::seconds(2),
                                                 boost::bind(&this_type::is_done,this)));
                HPX_TEST(!locked);
            }

            lock.unlock();
            t.join();
        }
        catch(...)
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
    typedef boost::unique_lock<M> try_lock_for_type;

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
            boost::chrono::system_clock::time_point xt =
                boost::chrono::system_clock::now()
              + boost::chrono::milliseconds(10);

            try_lock_for_type lock(mutex, xt);
            HPX_TEST(lock ? true : false);
        }
        {
            try_lock_for_type lock(mutex, boost::defer_lock);
            HPX_TEST(!lock);
        }
        try_lock_for_type lock(mutex);
        HPX_TEST(lock ? true : false);

        // Construct and initialize an xtime for a fast time out.
        boost::chrono::system_clock::time_point timeout =
            boost::chrono::system_clock::now()
          + boost::chrono::milliseconds(100);

        // Test the lock and the mutex with condition variables.
        // No one is going to notify this condition variable.  We expect to
        // time out.
        HPX_TEST(!condition.wait_until(lock, timeout, fake_predicate));
        HPX_TEST(lock ? true : false);

        boost::chrono::system_clock::time_point const now =
            boost::chrono::system_clock::now();
        HPX_TEST_LTE(timeout - timeout_resolution, now);

        // Test the lock, unlock and timedlock methods.
        lock.unlock();
        HPX_TEST(!lock);
        lock.lock();
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);

        boost::chrono::system_clock::time_point target =
            boost::chrono::system_clock::now()
          + boost::chrono::milliseconds(100);
        HPX_TEST(lock.try_lock_until(target));
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);

        HPX_TEST(mutex.try_lock_for(boost::chrono::milliseconds(100)));
        mutex.unlock();

        HPX_TEST(lock.try_lock_for(boost::chrono::milliseconds(100)));
        HPX_TEST(lock ? true : false);
        lock.unlock();
        HPX_TEST(!lock);
    }
};

template <typename M>
struct test_recursive_lock
{
    typedef M mutex_type;
    typedef boost::unique_lock<M> lock_type;

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
using boost::program_options::variables_map;
using boost::program_options::options_description;

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
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
