//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// clang-format off
#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/timing.hpp>
#include <hpx/concurrency/concurrent_unordered_map.hpp>
#include <hpx/concurrency/concurrent_vector.hpp>
// clang-format on
#include <iostream>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Baseline: std::vector + std::mutex
template <typename T>
struct std_vector_mutex
{
    std::vector<T> vec;
    mutable std::mutex mtx;

    void push_back(T const& val)
    {
        std::lock_guard<std::mutex> l(mtx);
        vec.push_back(val);
    }

    T operator[](size_t idx) const
    {
        std::lock_guard<std::mutex> l(mtx);
        return vec[idx];
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> l(mtx);
        return vec.size();
    }
};

// Baseline: std::unordered_map + std::mutex
template <typename Key, typename Val>
struct std_map_mutex
{
    std::unordered_map<Key, Val> map;
    mutable std::mutex mtx;

    void insert(std::pair<Key, Val> const& val)
    {
        std::lock_guard<std::mutex> l(mtx);
        map.insert(val);
    }

    Val operator[](Key const& key)
    {
        std::lock_guard<std::mutex> l(mtx);
        return map[key];
    }

    bool contains(Key const& key) const
    {
        std::lock_guard<std::mutex> l(mtx);
        return map.find(key) != map.end();
    }
};

///////////////////////////////////////////////////////////////////////////////
// Baseline: std::vector + hpx::spinlock (Simulating lightweight lock)
template <typename T>
struct std_vector_spinlock
{
    std::vector<T> vec;
    mutable hpx::util::spinlock mtx;

    void push_back(T const& val)
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        vec.push_back(val);
    }

    T operator[](size_t idx) const
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        return vec[idx];
    }

    size_t size() const
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        return vec.size();
    }
};

template <typename Key, typename Val>
struct std_map_spinlock
{
    std::unordered_map<Key, Val> map;
    mutable hpx::util::spinlock mtx;

    void insert(std::pair<Key, Val> const& val)
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        map.insert(val);
    }

    Val operator[](Key const& key)
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        return map[key];
    }

    bool contains(Key const& key) const
    {
        std::lock_guard<hpx::util::spinlock> l(mtx);
        return map.find(key) != map.end();
    }
};

///////////////////////////////////////////////////////////////////////////////

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t num_threads = vm["threads"].as<std::uint64_t>();
    std::uint64_t num_ops =
        vm["ops"].as<std::uint64_t>();    // Total ops per test

    // Test Vector
    {
        std::cout << "\n--- Vector Benchmark (Total Ops: " << num_ops
                  << ", Threads: " << num_threads << ") ---\n";

        auto run_vector_test = [&](auto& vec, std::string name) {
            std::vector<hpx::thread> threads;
            std::atomic<int> start_flag{0};

            hpx::chrono::high_resolution_timer t;

            for (std::uint64_t i = 0; i < num_threads; ++i)
            {
                threads.emplace_back([&, i]() {
                    while (start_flag.load() == 0)
                        hpx::this_thread::yield();
                    for (std::uint64_t j = 0; j < num_ops / num_threads; ++j)
                    {
                        vec.push_back(j);
                    }
                });
            }

            start_flag.store(1);
            for (auto& th : threads)
                th.join();

            double elapsed = t.elapsed();
            std::cout << name << " Push Back: " << elapsed << " s ("
                      << (num_ops / elapsed / 1e6) << " Mops/s)\n";

            // Random Access Test
            threads.clear();
            start_flag.store(0);
            t.restart();

            for (std::uint64_t i = 0; i < num_threads; ++i)
            {
                threads.emplace_back([&, i]() {
                    while (start_flag.load() == 0)
                        hpx::this_thread::yield();
                    for (std::uint64_t j = 0; j < num_ops / num_threads; ++j)
                    {
                        auto volatile val = vec[j % vec.size()];    // Read
                        (void) val;
                    }
                });
            }

            start_flag.store(1);
            for (auto& th : threads)
                th.join();

            elapsed = t.elapsed();
            std::cout << name << " Random Access: " << elapsed << " s ("
                      << (num_ops / elapsed / 1e6) << " Mops/s)\n";
        };

        {
            std_vector_mutex<int> v;
            run_vector_test(v, "std::vector + mutex    ");
        }
        {
            std_vector_spinlock<int> v;
            run_vector_test(v, "std::vector + spinlock ");
        }
        {
            hpx::concurrent::concurrent_vector<int> v;
            run_vector_test(v, "hpx::concurrent_vector ");
        }
    }

    // Test Unordered Map
    {
        std::cout << "\n--- Unordered Map Benchmark (Total Ops: " << num_ops
                  << ", Threads: " << num_threads << ") ---\n";

        auto run_map_test = [&](auto& m, std::string name) {
            std::vector<hpx::thread> threads;
            std::atomic<int> start_flag{0};

            hpx::chrono::high_resolution_timer t;

            // Insert
            for (std::uint64_t i = 0; i < num_threads; ++i)
            {
                threads.emplace_back([&, i]() {
                    while (start_flag.load() == 0)
                        hpx::this_thread::yield();
                    for (std::uint64_t j = 0; j < num_ops / num_threads; ++j)
                    {
                        m.insert(
                            {(int) (i * (num_ops / num_threads) + j), (int) j});
                    }
                });
            }

            start_flag.store(1);
            for (auto& th : threads)
                th.join();

            double elapsed = t.elapsed();
            std::cout << name << " Insert: " << elapsed << " s ("
                      << (num_ops / elapsed / 1e6) << " Mops/s)\n";

            // Read / Lookup
            threads.clear();
            start_flag.store(0);
            t.restart();

            for (std::uint64_t i = 0; i < num_threads; ++i)
            {
                threads.emplace_back([&, i]() {
                    while (start_flag.load() == 0)
                        hpx::this_thread::yield();
                    for (std::uint64_t j = 0; j < num_ops / num_threads; ++j)
                    {
                        // Access via operator[]
                        int volatile val =
                            m[(int) (i * (num_ops / num_threads) + j)];
                        (void) val;
                    }
                });
            }

            start_flag.store(1);
            for (auto& th : threads)
                th.join();

            elapsed = t.elapsed();
            std::cout << name << " Lookup: " << elapsed << " s ("
                      << (num_ops / elapsed / 1e6) << " Mops/s)\n";
        };

        {
            std_map_mutex<int, int> m;
            run_map_test(m, "std::unordered_map + mutex    ");
        }
        {
            std_map_spinlock<int, int> m;
            run_map_test(m, "std::unordered_map + spinlock ");
        }
        {
            hpx::concurrent::concurrent_unordered_map<int, int> m;
            run_map_test(m, "hpx::concurrent_unordered_map ");
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("threads,t",
        hpx::program_options::value<std::uint64_t>()->default_value(4),
        "number of threads")("ops,n",
        hpx::program_options::value<std::uint64_t>()->default_value(100000),
        "number of operations");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(hpx_main, argc, argv, init_args);
}
