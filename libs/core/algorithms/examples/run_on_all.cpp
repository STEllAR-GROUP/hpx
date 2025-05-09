//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/experimental/run_on_all.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/synchronization.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

int main(int argc, char* argv[])
{
    int num_threads = 4;
    if (argc >= 2)
    {
        num_threads = std::atoi(argv[1]);
    }

    constexpr char const* delim =
        "================================================================\n";
    std::cout << delim;
    std::cout << "hpx::run_on_all (using " << num_threads << " threads)\n";
    std::cout << delim;

    std::cout << "std::hardware_concurrency()   = "
              << std::thread::hardware_concurrency() << "\n";
    std::cout << "hpx::get_num_worker_threads() = "
              << hpx::get_num_worker_threads() << "\n";
    std::cout << delim;

    hpx::mutex mtx;
    hpx::experimental::run_on_all(
        hpx::execution::par,    // use parallel execution policy
        num_threads,    // use num_threads concurrent threads to execute the lambda
        [&](std::size_t index, std::tuple<> const& reductions) {
            std::lock_guard l(mtx);
            std::cout << "Hello! I am thread " << index << " of "
                      << hpx::get_num_worker_threads() << "\n";
            std::cout << "My C++ std::thread id is "
                      << std::this_thread::get_id() << "\n";
        });

    std::cout << delim;

    return 0;
}
