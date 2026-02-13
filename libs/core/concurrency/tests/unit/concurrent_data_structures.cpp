//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

void test_concurrent_vector()
{
    hpx::concurrent::concurrent_vector<int> v;
    std::atomic<int> count{0};

    std::vector<hpx::thread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&v, &count, i] {
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
    HPX_TEST_EQ(count.load(), 1000);

    // Test accessor
    if (v.size() > 0)
    {
        v[0] = 999;
        HPX_TEST_EQ(static_cast<int>(v[0]), 999);

        // Test operator T&
        int& ref = v[0];
        HPX_TEST_EQ(ref, 999);
        ref = 888;
        HPX_TEST_EQ(static_cast<int>(v[0]), 888);
    }
}

void test_concurrent_vector_reserve()
{
    hpx::concurrent::concurrent_vector<int> v;
    v.reserve(500);
    HPX_TEST(v.capacity() >= 500);

    for (int i = 0; i < 500; ++i)
        v.push_back(i);
    HPX_TEST_EQ(static_cast<int>(v.size()), 500);
}

void test_concurrent_unordered_map()
{
    hpx::concurrent::concurrent_unordered_map<int, int> m;
    std::atomic<int> count{0};

    std::vector<hpx::thread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&m, &count, i] {
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
        threads.emplace_back([&m, i] {
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

struct TransparentStringHash
{
    using is_transparent = void;
    size_t operator()(std::string_view sv) const
    {
        return std::hash<std::string_view>{}(sv);
    }
    size_t operator()(std::string const& s) const
    {
        return std::hash<std::string>{}(s);
    }
    size_t operator()(char const* s) const
    {
        return std::hash<std::string_view>{}(s);
    }
};

void test_concurrent_unordered_map_extra()
{
    // Test constructors
    hpx::concurrent::concurrent_unordered_map<int, int> m1(100);
    HPX_TEST(m1.bucket_count() >= 100);

    // Test heterogeneous lookup with transparent hash
    using MapType = hpx::concurrent::concurrent_unordered_map<std::string, int,
        TransparentStringHash, std::equal_to<>>;

    MapType m_str;
    std::string key = "hello";
    m_str.insert({key, 42});

    std::string_view key_view = "hello";
    // This should work with our new overloads and transparent hash
    auto acc = m_str.find(key_view);
    HPX_TEST(!acc.empty());
    if (acc)
    {
        HPX_TEST_EQ(acc.get(), 42);
    }

    HPX_TEST(m_str.contains(key_view));

    // operator[] heterogeneous - not supported by std::unordered_map yet (C++20)
    // We test standard operator[]
    m_str[key] = 43;    // via accessor::operator=
    HPX_TEST_EQ(m_str[key].get(), 43);
}

void test_concurrent_unordered_set()
{
    hpx::concurrent::concurrent_unordered_set<int> s;
    std::atomic<int> count{0};

    std::vector<hpx::thread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&s, &count, i] {
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

void test_concurrent_unordered_set_extra()
{
    using SetType = hpx::concurrent::concurrent_unordered_set<std::string,
        TransparentStringHash, std::equal_to<>>;

    SetType s;
    std::string key = "world";
    s.insert(key);

    std::string_view key_view = "world";
    HPX_TEST(s.contains(key_view));

    auto acc = s.find(key_view);
    HPX_TEST(!acc.empty());
    if (acc)
    {
        HPX_TEST_EQ(acc.get(), "world");    // accessor to key
    }
}

void test_concurrent_queue()
{
    hpx::concurrent::concurrent_queue<int>
        q;    // Test default ctor too? No cap
    hpx::concurrent::concurrent_queue<int> q100(100);
    std::atomic<int> count{0};

    std::vector<hpx::thread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        threads.emplace_back([&q100, &count, i] {
            for (int j = 0; j < 100; ++j)
            {
                q100.push(i * 100 + j);
                count++;
            }
        });
    }

    for (auto& t : threads)
        t.join();

    HPX_TEST_EQ(count.load(), 1000);

    int val;
    int popped_count = 0;
    while (q100.pop(val))
    {
        popped_count++;
    }

    HPX_TEST_EQ(popped_count, 1000);
}

int hpx_main(hpx::program_options::variables_map&)
{
    test_concurrent_vector();
    test_concurrent_vector_reserve();

    test_concurrent_unordered_map();
    test_concurrent_unordered_map_extra();

    test_concurrent_unordered_set();
    test_concurrent_unordered_set_extra();

    test_concurrent_queue();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params init_args;
    return hpx::init(hpx_main, argc, argv, init_args);
}
