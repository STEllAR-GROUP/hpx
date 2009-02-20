//  Copyright (c) 2009 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/lockfree/deque.hpp>

#include <climits>
#include <iostream>
#include <memory>

#define BOOST_TEST_MODULE lockfree_tests
#include <boost/test/included/unit_test.hpp>
#include <boost/thread.hpp>

#include "test_helpers.hpp"

using namespace boost;

void fill_deque(boost::lockfree::deque<int>& d)
{
    d.push_right(1);
    d.push_right(2);
    d.push_left(2);
    d.push_right(3);
    d.push_left(4);
}

BOOST_AUTO_TEST_CASE(simple_deque_test_pop_right)
{
    boost::lockfree::deque<int> f(64);
    BOOST_CHECK(f.empty());

    fill_deque(f);

    int i = 0;

    BOOST_CHECK(f.pop_right(&i));
    BOOST_CHECK_EQUAL(i, 3);
    BOOST_CHECK(f.pop_right(&i));
    BOOST_CHECK_EQUAL(i, 2);
    BOOST_CHECK(f.pop_right(&i));
    BOOST_CHECK_EQUAL(i, 1);
    BOOST_CHECK(f.pop_right(&i));
    BOOST_CHECK_EQUAL(i, 2);
    BOOST_CHECK(f.pop_right(&i));
    BOOST_CHECK_EQUAL(i, 4);

    BOOST_CHECK(f.empty());
}

BOOST_AUTO_TEST_CASE(simple_deque_test_pop_left)
{
    boost::lockfree::deque<int> f(64);
    BOOST_CHECK(f.empty());

    fill_deque(f);

    int i = 0;

    BOOST_CHECK(f.pop_left(&i));
    BOOST_CHECK_EQUAL(i, 4);
    BOOST_CHECK(f.pop_left(&i));
    BOOST_CHECK_EQUAL(i, 2);
    BOOST_CHECK(f.pop_left(&i));
    BOOST_CHECK_EQUAL(i, 1);
    BOOST_CHECK(f.pop_left(&i));
    BOOST_CHECK_EQUAL(i, 2);
    BOOST_CHECK(f.pop_left(&i));
    BOOST_CHECK_EQUAL(i, 3);

    BOOST_CHECK(f.empty());
}

boost::lockfree::deque<int> sd;

boost::lockfree::atomic_int<long> deque_cnt;

static_hashed_set<long, (1<<16)> working_set;

const unsigned int nodes_per_thread = 2000000 /* 00 */;

const int reader_threads = 5;
const int writer_threads = 5;

void add_right(void)
{
    for (unsigned int i = 0; i != nodes_per_thread; ++i)
    {
        while(deque_cnt > 10000)
            thread::yield();

        int id = generate_id<long>();

        bool inserted = working_set.insert(id);
        BOOST_ASSERT(inserted);
        sd.push_right(id);

        ++deque_cnt;
    }
}

void add_left(void)
{
    for (unsigned int i = 0; i != nodes_per_thread; ++i)
    {
        while(deque_cnt > 10000)
            thread::yield();

        int id = generate_id<long>();

        bool inserted = working_set.insert(id);
        BOOST_ASSERT(inserted);
        sd.push_left(id);

        ++deque_cnt;
    }
}

boost::lockfree::atomic_int<long> received_nodes;

bool get_element_right(void)
{
    int data;

    bool success = sd.pop_right(&data);
    if (success)
    {
        ++received_nodes;
        --deque_cnt;
        bool erased = working_set.erase(data);
        BOOST_ASSERT(erased);
        return true;
    }
    else
        return false;
}

bool get_element_left(void)
{
    int data;

    bool success = sd.pop_left(&data);
    if (success)
    {
        ++received_nodes;
        --deque_cnt;
        bool erased = working_set.erase(data);
        BOOST_ASSERT(erased);
        return true;
    }
    else
        return false;
}

volatile bool running = true;

void get_right(void)
{
    for(;;)
    {
        bool success = get_element_right();
        if (!running && !success)
            return;
        if (!success)
            thread::yield();
    }
}

void get_left(void)
{
    for(;;)
    {
        bool success = get_element_left();
        if (!running && !success)
            return;
        if (!success)
            thread::yield();
    }
}

BOOST_AUTO_TEST_CASE(deque_test_left)
{
    thread_group writer;
    thread_group reader;

    running = true;
    received_nodes = 0;

    for (int i = 0; i != reader_threads; ++i)
        reader.create_thread(&get_left);

    for (int i = 0; i != writer_threads; ++i)
        writer.create_thread(&add_left);
    std::cout << "deque_test_left: reader and writer threads created" << std::endl;

    writer.join_all();
    std::cout << "deque_test_left: writer threads joined. waiting for readers to finish" << std::endl;

    running = false;
    reader.join_all();

    BOOST_CHECK_EQUAL(received_nodes, writer_threads * nodes_per_thread);
    BOOST_CHECK_EQUAL(deque_cnt, 0);
    BOOST_CHECK(sd.empty());
    BOOST_CHECK(working_set.count_nodes() == 0);
}

BOOST_AUTO_TEST_CASE(deque_test_right)
{
    thread_group writer;
    thread_group reader;

    running = true;
    received_nodes = 0;

    for (int i = 0; i != reader_threads; ++i)
        reader.create_thread(&get_right);

    for (int i = 0; i != writer_threads; ++i)
        writer.create_thread(&add_right);
    std::cout << "deque_test_right: reader and writer threads created" << std::endl;

    writer.join_all();
    std::cout << "deque_test_right: writer threads joined. waiting for readers to finish" << std::endl;

    running = false;
    reader.join_all();

    BOOST_CHECK_EQUAL(received_nodes, writer_threads * nodes_per_thread);
    BOOST_CHECK_EQUAL(deque_cnt, 0);
    BOOST_CHECK(sd.empty());
    BOOST_CHECK(working_set.count_nodes() == 0);
}

BOOST_AUTO_TEST_CASE(deque_test_left_right)
{
    thread_group writer;
    thread_group reader;

    running = true;
    received_nodes = 0;

    for (int i = 0; i != reader_threads; ++i)
        reader.create_thread(&get_left);

    for (int i = 0; i != writer_threads; ++i)
        writer.create_thread(&add_right);
    std::cout << "deque_test_left_right: reader and writer threads created" << std::endl;

    writer.join_all();
    std::cout << "deque_test_left_right: writer threads joined. waiting for readers to finish" << std::endl;

    running = false;
    reader.join_all();

    BOOST_CHECK_EQUAL(received_nodes, writer_threads * nodes_per_thread);
    BOOST_CHECK_EQUAL(deque_cnt, 0);
    BOOST_CHECK(sd.empty());
    BOOST_CHECK(working_set.count_nodes() == 0);
}

BOOST_AUTO_TEST_CASE(deque_test_right_left)
{
    thread_group writer;
    thread_group reader;

    running = true;
    received_nodes = 0;

    for (int i = 0; i != reader_threads; ++i)
        reader.create_thread(&get_right);

    for (int i = 0; i != writer_threads; ++i)
        writer.create_thread(&add_left);
    std::cout << "deque_test_right_left: reader and writer threads created" << std::endl;

    writer.join_all();
    std::cout << "deque_test_right_left: writer threads joined. waiting for readers to finish" << std::endl;

    running = false;
    reader.join_all();

    BOOST_CHECK_EQUAL(received_nodes, writer_threads * nodes_per_thread);
    BOOST_CHECK_EQUAL(deque_cnt, 0);
    BOOST_CHECK(sd.empty());
    BOOST_CHECK(working_set.count_nodes() == 0);
}
