// (C) Copyright 2006-7 Anthony Williams
//  Copyright (c) 2015 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/apply.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/chrono.hpp>
#include <boost/thread/locks.hpp>

#include <mutex>

#include <string>
#include <vector>

#include "thread_group.hpp"
#include "shared_mutex_locking_thread.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value)           \
    {                                                                         \
        std::unique_lock<hpx::lcos::local::mutex> lock(mutex_name);         \
        HPX_TEST_EQ(value, expected_value);                                   \
    }

void test_multiple_readers()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    unsigned const number_of_threads = 10;

    test::thread_group pool;

    hpx::lcos::local::shared_mutex rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    try
    {
        for (unsigned i = 0; i != number_of_threads; ++i)
        {
            pool.create_thread(
                test::locking_thread<boost::shared_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running
                )
            );
        }

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while(unblocked_count < number_of_threads)
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, number_of_threads);

        finish_lock.unlock();
        pool.join_all();
    }
    catch(...)
    {
        pool.interrupt_all();
        pool.join_all();
        HPX_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, number_of_threads);
}

void test_only_one_writer_permitted()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    unsigned const number_of_threads = 10;

    test::thread_group pool;

    hpx::lcos::local::shared_mutex rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    try
    {
        for (unsigned i = 0; i != number_of_threads; ++i)
        {
            pool.create_thread(
                test::locking_thread<std::unique_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running
                )
            );
        }

        hpx::this_thread::sleep_for(boost::chrono::seconds(2));

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, 1u);

        finish_lock.unlock();
        pool.join_all();
    }
    catch(...)
    {
        pool.interrupt_all();
        pool.join_all();
        HPX_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        unblocked_count, number_of_threads);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, 1u);
}

void test_reader_blocks_writer()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    test::thread_group pool;

    hpx::lcos::local::shared_mutex rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running=0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    try
    {

        pool.create_thread(
            test::locking_thread<boost::shared_lock<shared_mutex_type> >(
                rw_mutex, unblocked_count, unblocked_count_mutex,
                unblocked_condition, finish_mutex,
                simultaneous_running_count, max_simultaneous_running
            )
        );

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while(unblocked_count<1)
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, 1u);

        pool.create_thread(
            test::locking_thread<std::unique_lock<shared_mutex_type> >(
                rw_mutex, unblocked_count, unblocked_count_mutex,
                unblocked_condition, finish_mutex,
                simultaneous_running_count, max_simultaneous_running
            )
        );

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, 1u);

        finish_lock.unlock();
        pool.join_all();
    }
    catch(...)
    {
        pool.interrupt_all();
        pool.join_all();
        HPX_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        unblocked_count, 2u);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, 1u);
}

void test_unlocking_writer_unblocks_all_readers()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    test::thread_group pool;

    hpx::lcos::local::shared_mutex rw_mutex;
    std::unique_lock<hpx::lcos::local::shared_mutex>  write_lock(rw_mutex);
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    std::unique_lock<mutex_type> finish_lock(finish_mutex);

    unsigned const reader_count = 10;

    try
    {
        for(unsigned i = 0; i != reader_count; ++i)
        {
            pool.create_thread(
                test::locking_thread<boost::shared_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running
                )
            );
        }

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, 0u);

        write_lock.unlock();

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while(unblocked_count<reader_count)
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, reader_count);

        finish_lock.unlock();
        pool.join_all();
    }
    catch(...)
    {
        pool.interrupt_all();
        pool.join_all();
        HPX_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, reader_count);
}

void test_unlocking_last_reader_only_unblocks_one_writer()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    test::thread_group pool;

    hpx::lcos::local::shared_mutex rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_readers = 0;
    unsigned max_simultaneous_readers = 0;
    unsigned simultaneous_running_writers = 0;
    unsigned max_simultaneous_writers = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_reading_mutex;
    std::unique_lock<mutex_type> finish_reading_lock(finish_reading_mutex);
    mutex_type finish_writing_mutex;
    std::unique_lock<mutex_type> finish_writing_lock(finish_writing_mutex);

    unsigned const reader_count = 10;
    unsigned const writer_count = 10;

    try
    {
        for (unsigned i = 0; i != reader_count; ++i)
        {
            pool.create_thread(
                test::locking_thread<boost::shared_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_reading_mutex,
                    simultaneous_running_readers, max_simultaneous_readers
                )
            );
        }

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));

        for(unsigned i = 0; i != writer_count; ++i)
        {
            pool.create_thread(
                test::locking_thread<std::unique_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_writing_mutex,
                    simultaneous_running_writers, max_simultaneous_writers
                )
            );
        }

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while(unblocked_count<reader_count)
            {
                unblocked_condition.wait(lk);
            }
        }

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, reader_count);

        finish_reading_lock.unlock();

        {
            std::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while (unblocked_count < (reader_count + 1))
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, reader_count + 1);

        finish_writing_lock.unlock();
        pool.join_all();
    }
    catch(...)
    {
        pool.interrupt_all();
        pool.join_all();
        HPX_TEST(false);
    }

    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        unblocked_count, reader_count + writer_count);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_readers, reader_count);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_writers, 1u);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_multiple_readers();
    test_only_one_writer_permitted();
    test_reader_blocks_writer();
    test_unlocking_writer_unblocks_all_readers();
    test_unlocking_last_reader_only_unblocks_one_writer();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
