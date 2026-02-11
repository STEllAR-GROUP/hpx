//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/concurrency/concurrent_queue.hpp>
#include <hpx/concurrency/concurrent_unordered_map.hpp>
#include <hpx/concurrency/concurrent_unordered_set.hpp>
#include <hpx/concurrency/concurrent_vector.hpp>
#include <hpx/concurrency/hash.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

void test_tbb_hash()
{
    hpx::concurrent::tbb_hash<int> h;
    int key = 42;
    size_t hash_val = h(key);
    HPX_TEST(hash_val != 0);    // Basic check

    hpx::concurrent::tbb_hasher<int> hasher;
    size_t hasher_val = hasher(key);
    HPX_TEST_EQ(hash_val, hasher_val);
}

void test_concurrent_vector()
{
    hpx::concurrent::concurrent_vector<int> v;
    std::atomic<int> count{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&v, &count, i]() {
            for (int j = 0; j < 100; ++j)
            {
                v.push_back(i * 100 + j);
                count++;
            }
        });
    }

    for (auto& t : threads)
        t.join();

    HPX_TEST_EQ(static_cast<int>(v.size()), 1000);
    HPX_TEST_EQ(static_cast<int>(v.size()), 1000);
    HPX_TEST_EQ(count.load(), 1000);

    // Test accessor
    if (v.size() > 0)
    {
        v[0] = 999;
        HPX_TEST_EQ(static_cast<int>(v[0]), 999);
    }
}

void test_concurrent_unordered_map()
{
    hpx::concurrent::concurrent_unordered_map<int, int> m;
    std::atomic<int> count{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&m, &count, i]() {
            for (int j = 0; j < 100; ++j)
            {
                m.insert({i * 100 + j, j});
                count++;
            }
        });
    }

    for (auto& t : threads)
        t.join();

    HPX_TEST_EQ(static_cast<int>(m.size()), 1000);
    HPX_TEST_EQ(count.load(), 1000);

    // Test concurrent read/write
    threads.clear();
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&m, i]() {
            for (int j = 0; j < 100; ++j)
            {
                m[i * 100 + j] = j + 1;      // Update
                int val = m[i * 100 + j];    // Read
                HPX_TEST_EQ(val, j + 1);
            }
        });
    }

    for (auto& t : threads)
        t.join();

    std::atomic<int> sum{0};
    m.for_each([&sum](auto const& kv) { sum += kv.second; });
    // 10 threads * sum(1..100) = 10 * 5050 = 50500
    HPX_TEST_EQ(sum.load(), 50500);
}

void test_concurrent_unordered_set()
{
    hpx::concurrent::concurrent_unordered_set<int> s;
    std::atomic<int> count{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&s, &count, i]() {
            for (int j = 0; j < 100; ++j)
            {
                s.insert(i * 100 + j);
                count++;
            }
        });
    }

    for (auto& t : threads)
        t.join();

    HPX_TEST_EQ(static_cast<int>(s.size()), 1000);
    HPX_TEST_EQ(count.load(), 1000);

    std::atomic<int> set_count{0};
    s.for_each([&set_count](auto const&) { set_count++; });
    HPX_TEST_EQ(set_count.load(), 1000);
}

void test_concurrent_queue()
{
    hpx::concurrent::concurrent_queue<int> q(100);
    std::atomic<int> count{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&q, &count, i]() {
            for (int j = 0; j < 100; ++j)
            {
                q.push(i * 100 + j);
                count++;
            }
        });
    }

    for (auto& t : threads)
        t.join();

    HPX_TEST_EQ(count.load(), 1000);

    int val;
    int popped_count = 0;
    while (q.pop(val))
    {
        popped_count++;
    }

    HPX_TEST_EQ(popped_count, 1000);
}

int main()
{
    test_concurrent_vector();
    test_concurrent_unordered_map();
    test_concurrent_unordered_set();
    test_concurrent_queue();
    test_tbb_hash();

    return hpx::util::report_errors();
}
