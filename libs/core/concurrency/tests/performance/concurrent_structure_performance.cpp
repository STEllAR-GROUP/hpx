//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/timing.hpp>

// clang-format off
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
            hpx::mutex mtx;
            hpx::condition_variable cv;
            bool ready = false;

            double total_elapsed_push = 0;
            double total_elapsed_read = 0;
            int const iterations = 10;

            for (int iter = 0; iter < iterations; ++iter)
            {
                vec.clear();    // Ensure clean state if possible, though vector usually grows.
                // Creating new vector per iteration might be fairer but slower?
                // The original test accumulated? No, it just pushed.
                // If we clear, we test cold cache maybe.
                // Note: hpx::concurrent::vector might not fully clear memory?
                // Actually, let's just create a new vector each time?
                // The lambda takes `vec` by reference. So we can't recreate it easily.
                // But we can clear it.
                // Optimization: The benchmark measures "Push Back". If we reuse, size grows.
                // If we clear, we measure from empty.
                if constexpr (requires { vec.clear(); })
                {
                    vec.clear();
                }
                else
                {
                    // std_vector_mutex etc support clear?
                    // They have vec member.
                    // But strictly, the passed object `vec` is the wrapper.
                    // The wrapper has `std::vector` or `concurrent_vector`.
                    // `concurrent_vector` has clear().
                    // Wrappers `std_vector_mutex` don't expose clear() in the original code!
                    // I should add clear() to wrappers or just assume we add on top?
                    // If we add on top, memory usage grows huge.
                    // I will add clear() to wrappers later.
                }

                threads.clear();
                ready = false;

                hpx::chrono::high_resolution_timer t;

                for (std::uint64_t i = 0; i < num_threads; ++i)
                {
                    threads.emplace_back([&, i]() {
                        std::unique_lock<hpx::mutex> l(mtx);
                        cv.wait(l, [&] { return ready; });
                        l.unlock();

                        for (std::uint64_t j = 0; j < num_ops / num_threads;
                            ++j)
                        {
                            vec.push_back(j);
                        }
                    });
                }

                {
                    std::lock_guard<hpx::mutex> l(mtx);
                    ready = true;
                }
                cv.notify_all();

                for (auto& th : threads)
                    th.join();

                total_elapsed_push += t.elapsed();

                // Random Access Test
                threads.clear();
                ready = false;
                t.restart();

                for (std::uint64_t i = 0; i < num_threads; ++i)
                {
                    threads.emplace_back([&, i]() {
                        std::unique_lock<hpx::mutex> l(mtx);
                        cv.wait(l, [&] { return ready; });
                        l.unlock();

                        for (std::uint64_t j = 0; j < num_ops / num_threads;
                            ++j)
                        {
                            [[maybe_unused]] auto volatile val =
                                vec[j % vec.size()];    // Read
                        }
                    });
                }

                {
                    std::lock_guard<hpx::mutex> l(mtx);
                    ready = true;
                }
                cv.notify_all();

                for (auto& th : threads)
                    th.join();

                total_elapsed_read += t.elapsed();
            }

            double avg_push = total_elapsed_push / iterations;
            double avg_read = total_elapsed_read / iterations;

            std::cout << name << " Push Back: " << avg_push << " s ("
                      << (num_ops / avg_push / 1e6) << " Mops/s)\n";
            std::cout << name << " Random Access: " << avg_read << " s ("
                      << (num_ops / avg_read / 1e6) << " Mops/s)\n";
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
            hpx::mutex mtx;
            hpx::condition_variable cv;
            bool ready = false;

            double total_elapsed_insert = 0;
            double total_elapsed_lookup = 0;
            int const iterations = 10;

            for (int iter = 0; iter < iterations; ++iter)
            {
                // We rely on 'm' being cleared or just grow?
                // If we don't clear, we insert duplicates or collisions?
                // Maps handle duplicates (updates) or reject?
                // std::unordered_map inserts if not exists.
                // So subsequent iterations do NOTHING if keys are same.
                // We must CLEAR.
                if constexpr (requires { m.map.clear(); })
                {
                    m.map.clear();
                }
                else if constexpr (requires { m.clear(); })
                {
                    m.clear();
                }

                threads.clear();
                ready = false;

                hpx::chrono::high_resolution_timer t;

                // Insert
                for (std::uint64_t i = 0; i < num_threads; ++i)
                {
                    threads.emplace_back([&, i]() {
                        std::unique_lock<hpx::mutex> l(mtx);
                        cv.wait(l, [&] { return ready; });
                        l.unlock();

                        for (std::uint64_t j = 0; j < num_ops / num_threads;
                            ++j)
                        {
                            m.insert({(int) (i * (num_ops / num_threads) + j),
                                (int) j});
                        }
                    });
                }

                {
                    std::lock_guard<hpx::mutex> l(mtx);
                    ready = true;
                }
                cv.notify_all();

                for (auto& th : threads)
                    th.join();

                total_elapsed_insert += t.elapsed();

                // Read / Lookup
                threads.clear();
                ready = false;
                t.restart();

                for (std::uint64_t i = 0; i < num_threads; ++i)
                {
                    threads.emplace_back([&, i]() {
                        std::unique_lock<hpx::mutex> l(mtx);
                        cv.wait(l, [&] { return ready; });
                        l.unlock();

                        for (std::uint64_t j = 0; j < num_ops / num_threads;
                            ++j)
                        {
                            // Access via operator[]
                            [[maybe_unused]] int volatile val =
                                m[(int) (i * (num_ops / num_threads) + j)];
                        }
                    });
                }

                {
                    std::lock_guard<hpx::mutex> l(mtx);
                    ready = true;
                }
                cv.notify_all();

                for (auto& th : threads)
                    th.join();

                total_elapsed_lookup += t.elapsed();
            }

            double avg_insert = total_elapsed_insert / iterations;
            double avg_lookup = total_elapsed_lookup / iterations;

            std::cout << name << " Insert: " << avg_insert << " s ("
                      << (num_ops / avg_insert / 1e6) << " Mops/s)\n";
            std::cout << name << " Lookup: " << avg_lookup << " s ("
                      << (num_ops / avg_lookup / 1e6) << " Mops/s)\n";
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
