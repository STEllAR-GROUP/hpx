//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#include <hpx/barrier.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

std::atomic<std::size_t> c1(0);
std::atomic<std::size_t> c2(0);

///////////////////////////////////////////////////////////////////////////////
void local_barrier_test_no_completion(hpx::barrier<>& b)
{
    ++c1;

    // wait for all threads to enter the barrier
    b.arrive_and_wait();

    ++c2;
}

void test_barrier_empty_oncomplete()
{
    constexpr std::size_t threads = 64;
    constexpr std::size_t iterations = 100;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        hpx::barrier<> b(threads + 1);
        c1 = 0;
        c2 = 0;

        // create the threads which will wait on the barrier
        std::vector<hpx::future<void>> results;
        results.reserve(threads);
        for (std::size_t i = 0; i != threads; ++i)
        {
            results.push_back(
                hpx::async(&local_barrier_test_no_completion, std::ref(b)));
        }

        b.arrive_and_wait();    // wait for all threads to enter the barrier
        HPX_TEST_EQ(threads, c1);

        hpx::wait_all(results);

        HPX_TEST_EQ(threads, c2);
    }
}

///////////////////////////////////////////////////////////////////////////////
std::atomic<std::size_t> complete(0);

struct oncomplete
{
    void operator()() const
    {
        ++complete;
    }
};

void local_barrier_test(hpx::barrier<oncomplete>& b)
{
    ++c1;

    // wait for all threads to enter the barrier
    b.arrive_and_wait();

    ++c2;
}

void test_barrier_oncomplete()
{
    constexpr std::size_t threads = 64;
    constexpr std::size_t iterations = 100;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        hpx::barrier<oncomplete> b(threads + 1);
        c1 = 0;
        c2 = 0;
        complete = 0;

        // create the threads which will wait on the barrier
        std::vector<hpx::future<void>> results;
        results.reserve(threads);
        for (std::size_t i = 0; i != threads; ++i)
        {
            results.push_back(hpx::async(&local_barrier_test, std::ref(b)));
        }

        b.arrive_and_wait();    // wait for all threads to enter the barrier
        HPX_TEST_EQ(threads, c1);

        hpx::wait_all(results);

        HPX_TEST_EQ(threads, c2);
        HPX_TEST_EQ(complete, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
void local_barrier_test_no_completion_split(hpx::barrier<>& b)
{
    // signal the barrier
    auto token = b.arrive();

    ++c1;

    // wait for all threads to enter the barrier
    b.wait(std::move(token));

    ++c2;
}

void test_barrier_empty_oncomplete_split()
{
    constexpr std::size_t threads = 64;
    constexpr std::size_t iterations = 100;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        hpx::barrier<> b(threads + 1);
        c1 = 0;
        c2 = 0;

        // create the threads which will wait on the barrier
        std::vector<hpx::future<void>> results;
        results.reserve(threads);
        for (std::size_t i = 0; i != threads; ++i)
        {
            results.push_back(hpx::async(
                &local_barrier_test_no_completion_split, std::ref(b)));
        }

        b.arrive_and_wait();    // wait for all threads to enter the barrier
        HPX_TEST_EQ(threads, c1);

        hpx::wait_all(results);

        HPX_TEST_EQ(threads, c2);
    }
}

void local_barrier_test_split(hpx::barrier<oncomplete>& b)
{
    // signal the barrier
    auto token = b.arrive();

    ++c1;

    // wait for all threads to enter the barrier
    b.wait(std::move(token));

    ++c2;
}

void test_barrier_oncomplete_split()
{
    constexpr std::size_t threads = 64;
    constexpr std::size_t iterations = 100;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        // create a barrier waiting on 'count' threads
        hpx::barrier<oncomplete> b(threads + 1);
        c1 = 0;
        c2 = 0;
        complete = 0;

        // create the threads which will wait on the barrier
        std::vector<hpx::future<void>> results;
        results.reserve(threads);
        for (std::size_t i = 0; i != threads; ++i)
        {
            results.push_back(
                hpx::async(&local_barrier_test_split, std::ref(b)));
        }

        b.arrive_and_wait();    // wait for all threads to enter the barrier
        HPX_TEST_EQ(threads, c1);

        hpx::wait_all(results);

        HPX_TEST_EQ(threads, c2);
        HPX_TEST_EQ(complete, std::size_t(1));
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_barrier_empty_oncomplete();
    test_barrier_oncomplete();

    test_barrier_empty_oncomplete_split();
    test_barrier_oncomplete_split();

    return 0;
}
