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

#include "thread_group.hpp"
#include "shared_mutex_locking_thread.hpp"

#define CHECK_LOCKED_VALUE_EQUAL(mutex_name, value, expected_value)           \
    {                                                                         \
        boost::unique_lock<hpx::lcos::local::mutex> lock(mutex_name);         \
        HPX_TEST_EQ(value, expected_value);                                   \
    }

void test_only_one_upgrade_lock_permitted()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    unsigned const number_of_threads = 2;

    test::thread_group pool;

    shared_mutex_type rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);

    try
    {
        for( unsigned i = 0; i != number_of_threads; ++i)
        {
            pool.create_thread(
                test::locking_thread<boost::upgrade_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count, unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running
                )
            );
        }

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
        unblocked_count, number_of_threads);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, 1u);
}

void test_can_lock_upgrade_if_currently_locked_shared()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    test::thread_group pool;

    shared_mutex_type rw_mutex;
    unsigned unblocked_count = 0;
    unsigned simultaneous_running_count = 0;
    unsigned max_simultaneous_running = 0;
    mutex_type unblocked_count_mutex;
    hpx::lcos::local::condition_variable unblocked_condition;
    mutex_type finish_mutex;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);

    unsigned const reader_count = 10;

    try
    {
        for(unsigned i = 0; i != reader_count; ++i)
        {
            pool.create_thread(
                test::locking_thread<boost::shared_lock<shared_mutex_type> >(
                    rw_mutex, unblocked_count,  unblocked_count_mutex,
                    unblocked_condition, finish_mutex,
                    simultaneous_running_count, max_simultaneous_running
                )
            );
        }

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));

        pool.create_thread(
            test::locking_thread<boost::upgrade_lock<shared_mutex_type> >(
                rw_mutex, unblocked_count, unblocked_count_mutex,
                unblocked_condition, finish_mutex,
                simultaneous_running_count, max_simultaneous_running
            )
        );

        {
            boost::unique_lock<mutex_type> lk(unblocked_count_mutex);
            while(unblocked_count < (reader_count + 1))
            {
                unblocked_condition.wait(lk);
            }
        }

        CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
            unblocked_count, reader_count + 1);

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
        unblocked_count, reader_count + 1);
    CHECK_LOCKED_VALUE_EQUAL(unblocked_count_mutex,
        max_simultaneous_running, reader_count + 1);
}

void test_can_lock_upgrade_to_unique_if_currently_locked_upgrade()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type mtx;
    boost::upgrade_lock<shared_mutex_type> l(mtx);
    boost::upgrade_to_unique_lock<shared_mutex_type> ul(l);
    HPX_TEST(ul.owns_lock());
}

void test_if_other_thread_has_write_lock_try_lock_shared_returns_false()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);
    hpx::thread writer(test::simple_writing_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    hpx::this_thread::sleep_for(boost::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex,
        unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_shared();
    HPX_TEST(!try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_other_thread_has_write_lock_try_lock_upgrade_returns_false()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);
    hpx::thread writer(test::simple_writing_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    hpx::this_thread::sleep_for(boost::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex,
        unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    HPX_TEST(!try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_no_thread_has_lock_try_lock_shared_returns_true()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type rw_mutex;
    bool const try_succeeded = rw_mutex.try_lock_shared();
    HPX_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }
}

void test_if_no_thread_has_lock_try_lock_upgrade_returns_true()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;

    shared_mutex_type rw_mutex;
    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    HPX_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }
}

void test_if_other_thread_has_shared_lock_try_lock_shared_returns_true()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);
    hpx::thread writer(test::simple_reading_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    hpx::this_thread::sleep_for(boost::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex,
        unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_shared();
    HPX_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_shared();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_other_thread_has_shared_lock_try_lock_upgrade_returns_true()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);
    hpx::thread writer(test::simple_reading_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    hpx::this_thread::sleep_for(boost::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex,
        unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    HPX_TEST(try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }

    finish_lock.unlock();
    writer.join();
}

void test_if_other_thread_has_upgrade_lock_try_lock_upgrade_returns_false()
{
    typedef hpx::lcos::local::shared_mutex shared_mutex_type;
    typedef hpx::lcos::local::mutex mutex_type;

    shared_mutex_type rw_mutex;
    mutex_type finish_mutex;
    mutex_type unblocked_mutex;
    unsigned unblocked_count = 0;
    boost::unique_lock<mutex_type> finish_lock(finish_mutex);
    hpx::thread writer(test::simple_upgrade_thread(
        rw_mutex, finish_mutex, unblocked_mutex, unblocked_count));

    hpx::this_thread::sleep_for(boost::chrono::seconds(1));

    CHECK_LOCKED_VALUE_EQUAL(unblocked_mutex,
        unblocked_count, 1u);

    bool const try_succeeded = rw_mutex.try_lock_upgrade();
    HPX_TEST(!try_succeeded);
    if (try_succeeded)
    {
        rw_mutex.unlock_upgrade();
    }

    finish_lock.unlock();
    writer.join();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_only_one_upgrade_lock_permitted();
    test_can_lock_upgrade_if_currently_locked_shared();
    test_can_lock_upgrade_to_unique_if_currently_locked_upgrade();
    test_if_other_thread_has_write_lock_try_lock_shared_returns_false();
    test_if_other_thread_has_write_lock_try_lock_upgrade_returns_false();
    test_if_no_thread_has_lock_try_lock_shared_returns_true();
    test_if_no_thread_has_lock_try_lock_upgrade_returns_true();
    test_if_other_thread_has_shared_lock_try_lock_shared_returns_true();
    test_if_other_thread_has_shared_lock_try_lock_upgrade_returns_true();
    test_if_other_thread_has_upgrade_lock_try_lock_upgrade_returns_false();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

