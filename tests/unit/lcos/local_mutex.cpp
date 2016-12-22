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
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/shared_spinlock.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>


#include <chrono>
#include <mutex>
#include <shared_mutex>
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
            std::chrono::system_clock::now()
          + std::chrono::milliseconds(10);

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
            std::chrono::system_clock::now()
          + std::chrono::milliseconds(10);

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
    typedef std::unique_lock<Mutex> Lock;

    Mutex m;
    hpx::lcos::local::spinlock done_mutex;
    bool done;
    bool locked;
    hpx::lcos::local::condition_variable_any done_cond;

    test_lock_times_out_if_other_thread_has_lock():
        done(false),locked(false)
    {}

    void locking_thread()
    {
        Lock lock(m,std::defer_lock);
        lock.try_lock_for(std::chrono::milliseconds(50));

        std::lock_guard<hpx::lcos::local::spinlock> lk(done_mutex);
        locked=lock.owns_lock();
        done=true;
        done_cond.notify_one();
    }

    void locking_thread_through_constructor()
    {
        Lock lock(m,std::chrono::milliseconds(50));

        std::lock_guard<hpx::lcos::local::spinlock> lk(done_mutex);
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
                std::unique_lock<hpx::lcos::local::spinlock> lk(done_mutex);
                HPX_TEST(done_cond.wait_for(lk,std::chrono::seconds(2),
                    hpx::util::bind(&this_type::is_done,this)));
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
                std::chrono::system_clock::now()
              + std::chrono::milliseconds(10);

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
            std::chrono::system_clock::now()
          + std::chrono::milliseconds(100);

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
            std::chrono::system_clock::now()
          + std::chrono::milliseconds(100);
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

template <typename M, typename L>
struct shared_locking_thread
{
    typedef M shared_mutex_type;

    shared_mutex_type& rw_mutex;
    unsigned& unblocked_count;
    hpx::lcos::local::condition_variable_any& unblocked_condition;
    unsigned& simultaneous_running_count;
    unsigned& max_simultaneous_running;
    hpx::lcos::local::spinlock& unblocked_count_mutex;
    hpx::lcos::local::spinlock& finish_mutex;

    shared_locking_thread(shared_mutex_type& rw_mutex_,
        unsigned& unblocked_count_,
        hpx::lcos::local::condition_variable_any& unblocked_condition_,
        unsigned& simultaneous_running_count_,
        unsigned& max_simultaneous_running_,
        hpx::lcos::local::spinlock& unblocked_count_mutex_,
        hpx::lcos::local::spinlock& finish_mutex_)
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
        // acquire_lock
        L lock(rw_mutex);
        hpx::util::ignore_while_checking<L> il(&lock);

        // increment count to show we're unblocked
        {
            std::unique_lock<hpx::lcos::local::spinlock> ublock(unblocked_count_mutex);
            ++unblocked_count;
            ++simultaneous_running_count;
            if (simultaneous_running_count > max_simultaneous_running)
            {
                max_simultaneous_running = simultaneous_running_count;
            }
        }
        unblocked_condition.notify_one();

        // wait to finish
        std::unique_lock<hpx::lcos::local::spinlock> finish_lock(finish_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> ill(&finish_lock);
        {
            std::unique_lock<hpx::lcos::local::spinlock> ublock(unblocked_count_mutex);

            --simultaneous_running_count;
        }
    }

};

template <typename M>
struct test_shared_lock
{
    typedef M mutex_type;

    void test_multiple_readers()
    {
        mutex_type rw_mutex;
        unsigned unblocked_count = 0;
        unsigned simultaneous_running_count = 0;
        unsigned max_simultaneous_running = 0;
        hpx::lcos::local::spinlock unblocked_count_mutex;
        hpx::lcos::local::condition_variable_any unblocked_condition;
        hpx::lcos::local::spinlock finish_mutex;

        std::unique_lock<hpx::lcos::local::spinlock> finish_lock(finish_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> il(&finish_lock);

        unsigned const number_of_threads = 10;

        std::vector<hpx::thread> threads;
        for (unsigned i = 0; i < number_of_threads; ++i)
        {
            threads.emplace_back(
                shared_locking_thread<mutex_type, std::shared_lock<mutex_type>>(
                    rw_mutex,
                    unblocked_count,
                    unblocked_condition,
                    simultaneous_running_count,
                    max_simultaneous_running,
                    unblocked_count_mutex,
                    finish_mutex
                )
            );
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            while (unblocked_count < number_of_threads)
            {
                unblocked_condition.wait(lk);
            }
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(max_simultaneous_running, number_of_threads);
        }

        finish_lock.unlock();

        for (auto& t: threads)
        {
            t.join();
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(max_simultaneous_running, number_of_threads);
        }
    }

    void test_only_one_writer_permitted()
    {
        mutex_type rw_mutex;
        unsigned unblocked_count = 0;
        unsigned simultaneous_running_count = 0;
        unsigned max_simultaneous_running = 0;
        hpx::lcos::local::spinlock unblocked_count_mutex;
        hpx::lcos::local::condition_variable_any unblocked_condition;
        hpx::lcos::local::spinlock finish_mutex;

        std::unique_lock<hpx::lcos::local::spinlock> finish_lock(finish_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> il(&finish_lock);

        unsigned const number_of_threads = 10;

        std::vector<hpx::thread> threads;
        for (unsigned i = 0; i < number_of_threads; ++i)
        {
            threads.emplace_back(
                shared_locking_thread<mutex_type, std::unique_lock<mutex_type>>(
                    rw_mutex,
                    unblocked_count,
                    unblocked_condition,
                    simultaneous_running_count,
                    max_simultaneous_running,
                    unblocked_count_mutex,
                    finish_mutex
                )
            );
        }

        hpx::this_thread::sleep_for(std::chrono::microseconds(2));

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, 1u);
        }

        finish_lock.unlock();

        for (auto& t: threads)
        {
            t.join();
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, number_of_threads);
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(max_simultaneous_running, 1u);
        }
    }

    void test_reader_blocks_writer()
    {
        mutex_type rw_mutex;
        unsigned unblocked_count = 0;
        unsigned simultaneous_running_count = 0;
        unsigned max_simultaneous_running = 0;
        hpx::lcos::local::spinlock unblocked_count_mutex;
        hpx::lcos::local::condition_variable_any unblocked_condition;
        hpx::lcos::local::spinlock finish_mutex;

        std::unique_lock<hpx::lcos::local::spinlock> finish_lock(finish_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> il(&finish_lock);

        std::vector<hpx::thread> threads;

        threads.emplace_back(
            shared_locking_thread<mutex_type, std::shared_lock<mutex_type>>(
                rw_mutex,
                unblocked_count,
                unblocked_condition,
                simultaneous_running_count,
                max_simultaneous_running,
                unblocked_count_mutex,
                finish_mutex
            )
        );

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            while (unblocked_count < 1)
            {
                unblocked_condition.wait(lk);
            }
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, 1u);
        }

        threads.emplace_back(
            shared_locking_thread<mutex_type, std::unique_lock<mutex_type>>(
                rw_mutex,
                unblocked_count,
                unblocked_condition,
                simultaneous_running_count,
                max_simultaneous_running,
                unblocked_count_mutex,
                finish_mutex
            )
        );

        hpx::this_thread::sleep_for(std::chrono::microseconds(2));
        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, 1u);
        }

        finish_lock.unlock();

        for (auto& t: threads)
        {
            t.join();
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, 2u);
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(max_simultaneous_running, 1u);
        }
    }

    void test_unlocking_writer_unblocks_all_readers()
    {
        mutex_type rw_mutex;
        std::unique_lock<mutex_type> write_lock(rw_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<mutex_type>> iwl(&write_lock);
        unsigned unblocked_count = 0;
        unsigned simultaneous_running_count = 0;
        unsigned max_simultaneous_running = 0;
        hpx::lcos::local::spinlock unblocked_count_mutex;
        hpx::lcos::local::condition_variable_any unblocked_condition;
        hpx::lcos::local::spinlock finish_mutex;

        std::unique_lock<hpx::lcos::local::spinlock> finish_lock(finish_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> il(&finish_lock);

        std::vector<hpx::thread> threads;

        unsigned const reader_count=10;

        for (unsigned i = 0; i < reader_count; ++i)
        {
            threads.emplace_back(
                shared_locking_thread<mutex_type, std::shared_lock<mutex_type>>(
                    rw_mutex,
                    unblocked_count,
                    unblocked_condition,
                    simultaneous_running_count,
                    max_simultaneous_running,
                    unblocked_count_mutex,
                    finish_mutex
                )
            );
        }
        hpx::this_thread::sleep_for(std::chrono::microseconds(2));

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, 0u);
        }

        write_lock.unlock();

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            while (unblocked_count < reader_count)
            {
                unblocked_condition.wait(lk);
            }
            HPX_TEST_EQ(unblocked_count, reader_count);
        }
        finish_lock.unlock();

        for (auto& t: threads)
        {
            t.join();
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(max_simultaneous_running, 1u);
        }
    }

    void test_unlocking_last_reader_only_unblocks_one_writer()
    {
        mutex_type rw_mutex;
        unsigned unblocked_count = 0;
        unsigned simultaneous_running_readers = 0;
        unsigned simultaneous_running_writers = 0;
        unsigned max_simultaneous_readers = 0;
        unsigned max_simultaneous_writers = 0;
        hpx::lcos::local::spinlock unblocked_count_mutex;
        hpx::lcos::local::condition_variable_any unblocked_condition;
        hpx::lcos::local::spinlock finish_reading_mutex;
        hpx::lcos::local::spinlock finish_writing_mutex;

        std::unique_lock<hpx::lcos::local::spinlock> finish_reading_lock(finish_reading_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> irl(&finish_reading_lock);

        std::unique_lock<hpx::lcos::local::spinlock> finish_writing_lock(finish_writing_mutex);
        hpx::util::ignore_while_checking<std::unique_lock<hpx::lcos::local::spinlock>> iwl(&finish_writing_lock);

        std::vector<hpx::thread> threads;

        unsigned const reader_count = 10;
        unsigned const writer_count = 10;

        for (unsigned i = 0; i < reader_count; ++i)
        {
            threads.emplace_back(
                shared_locking_thread<mutex_type, std::shared_lock<mutex_type>>(
                    rw_mutex,
                    unblocked_count,
                    unblocked_condition,
                    simultaneous_running_readers,
                    max_simultaneous_readers,
                    unblocked_count_mutex,
                    finish_reading_mutex
                )
            );
        }
        hpx::this_thread::sleep_for(std::chrono::microseconds(2));

        for (unsigned i = 0; i < reader_count; ++i)
        {
            threads.emplace_back(
                shared_locking_thread<mutex_type, std::unique_lock<mutex_type>>(
                    rw_mutex,
                    unblocked_count,
                    unblocked_condition,
                    simultaneous_running_writers,
                    max_simultaneous_writers,
                    unblocked_count_mutex,
                    finish_writing_mutex
                )
            );
        }
        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            while (unblocked_count < reader_count)
            {
                unblocked_condition.wait(lk);
            }
        }
        hpx::this_thread::sleep_for(std::chrono::microseconds(2));

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, reader_count);
        }

        finish_reading_lock.unlock();

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            while (unblocked_count < reader_count + 1)
            {
                unblocked_condition.wait(lk);
            }
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, reader_count);
        }

        finish_writing_lock.unlock();

        for (auto& t: threads)
        {
            t.join();
        }

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(unblocked_count_mutex);
            HPX_TEST_EQ(unblocked_count, reader_count + writer_count);
            HPX_TEST_EQ(max_simultaneous_readers, reader_count);
            HPX_TEST_EQ(max_simultaneous_writers, 1u);
        }
    }

    void operator()()
    {
        test_multiple_readers();
        test_only_one_writer_permitted();
        test_reader_blocks_writer();
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

void test_spinlock()
{
    test_lock<hpx::lcos::local::spinlock>()();
    test_trylock<hpx::lcos::local::spinlock>()();
}

void test_shared_spinlock()
{
    test_lock<hpx::lcos::local::shared_spinlock>()();
    test_trylock<hpx::lcos::local::shared_spinlock>()();
    test_shared_lock<hpx::lcos::local::shared_spinlock>()();
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
        test_spinlock();
        test_shared_spinlock();
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
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
