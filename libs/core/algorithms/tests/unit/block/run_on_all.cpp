//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/experimental/run_on_all.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

int main()
{
    using namespace hpx::experimental;

    // Test basic functionality with reduction
    {
        std::uint32_t n = 0;
        run_on_all(
            reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(
            n, static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
    }

    // Test with specific number of tasks
    {
        std::uint32_t n = 0;
        run_on_all(
            2, reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(n, static_cast<std::uint32_t>(2));
    }

    // Test with sequential execution policy
    {
        std::uint32_t n = 0;
        run_on_all(hpx::execution::seq, reduction_plus(n),
            [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(
            n, static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
    }

    // Test with parallel execution policy
    {
        std::uint32_t n = 0;
        run_on_all(hpx::execution::par, reduction_plus(n),
            [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(
            n, static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
    }

    // Test with parallel unsequenced execution policy
    {
        std::uint32_t n = 0;
        run_on_all(hpx::execution::par_unseq, reduction_plus(n),
            [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(
            n, static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
    }

    // Test with multiple arguments
    {
        std::uint32_t n = 0;
        std::uint32_t m = 0;
        run_on_all(reduction_plus(n), reduction_plus(m),
            [](std::uint32_t& local_n, std::uint32_t& local_m) {
                ++local_n;
                local_m += 2;
            });
        HPX_TEST_EQ(
            n, static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
        HPX_TEST_EQ(
            m, static_cast<std::uint32_t>(2 * hpx::get_num_worker_threads()));
    }

    // Test with vector reduction
    {
        std::vector<std::uint32_t> v(hpx::get_num_worker_threads(), 0);
        run_on_all(reduction_plus(v), [](std::vector<std::uint32_t>& local_v) {
            local_v[hpx::get_worker_thread_num()] = 1;
        });
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            HPX_TEST_EQ(v[i], static_cast<std::uint32_t>(1));
        }
    }

    // Test with atomic operations
    {
        std::atomic<std::uint32_t> n(0);
        run_on_all([&]() { ++n; });
        HPX_TEST_EQ(n.load(),
            static_cast<std::uint32_t>(hpx::get_num_worker_threads()));
    }

    // Test with different number of tasks
    {
        std::uint32_t n = 0;
        run_on_all(
            1, reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(n, static_cast<std::uint32_t>(1));

        n = 0;
        run_on_all(
            4, reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(n, static_cast<std::uint32_t>(4));
    }

    return hpx::util::report_errors();
}
