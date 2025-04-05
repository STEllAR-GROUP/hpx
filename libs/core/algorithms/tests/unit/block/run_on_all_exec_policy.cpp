//  Copyright (c) 2024 Harith
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/run_on_all.hpp>
#include <hpx/testing.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

int hpx_main()
{
    {
        // Test synchronous execution
        std::atomic<std::size_t> count{0};
        auto result = hpx::parallel::run_on_all(
            hpx::execution::seq, 0, [&count](std::size_t) { ++count; });

        HPX_TEST_EQ(count.load(), hpx::get_num_localities(hpx::launch::sync));
    }

    {
        // Test asynchronous execution
        std::atomic<std::size_t> count{0};
        auto future = hpx::parallel::run_on_all(
            hpx::execution::par(hpx::execution::task), 0,
            [&count](std::size_t) { ++count; });

        future.get();
        HPX_TEST_EQ(count.load(), hpx::get_num_localities(hpx::launch::sync));
    }

    {
        // Test with reduction
        std::vector<std::size_t> values(hpx::get_num_localities(hpx::launch::sync));
        for (std::size_t i = 0; i != values.size(); ++i)
        {
            values[i] = i + 1;
        }

        auto result = hpx::parallel::run_on_all(hpx::execution::par, 0,
            [&values](std::size_t i) { return values[i]; },
            [](std::size_t a, std::size_t b) { return a + b; });

        std::size_t expected = 0;
        for (std::size_t i = 0; i != values.size(); ++i)
        {
            expected += values[i];
        }

        HPX_TEST_EQ(result, expected);
    }

    {
        // Test async with reduction
        std::vector<std::size_t> values(hpx::get_num_localities(hpx::launch::sync));
        for (std::size_t i = 0; i != values.size(); ++i)
        {
            values[i] = i + 1;
        }

        auto future = hpx::parallel::run_on_all(
            hpx::execution::par(hpx::execution::task), 0,
            [&values](std::size_t i) { return values[i]; },
            [](std::size_t a, std::size_t b) { return a + b; });

        std::size_t expected = 0;
        for (std::size_t i = 0; i != values.size(); ++i)
        {
            expected += values[i];
        }

        HPX_TEST_EQ(future.get(), expected);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
} 