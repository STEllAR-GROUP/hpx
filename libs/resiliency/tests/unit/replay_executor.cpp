//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

struct vogon_exception : std::exception
{
};

int hpx_main(int argc, char* argv[])
{
    try
    {
        hpx::parallel::execution::parallel_executor base_exec;
        auto exec =
            hpx::resiliency::experimental::make_replay_executor(base_exec, 3);

        std::vector<std::size_t> data(100);
        std::iota(data.begin(), data.end(), 0);

        std::atomic<std::size_t> count(0);
        hpx::parallel::for_each(hpx::parallel::execution::par.on(exec),
            data.begin(), data.end(), [&](std::size_t i) {
                if (++count == 42)
                {
                    throw vogon_exception();
                }
            });
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    try
    {
        hpx::parallel::execution::parallel_executor base_exec;
        auto exec =
            hpx::resiliency::experimental::make_replay_executor(base_exec, 3);

        std::vector<std::size_t> data(100);
        std::iota(data.begin(), data.end(), 0);

        std::vector<std::size_t> dest(100);

        std::atomic<std::size_t> count(0);
        hpx::parallel::transform(hpx::parallel::execution::par.on(exec),
            data.begin(), data.end(), dest.begin(), [&](std::size_t i) {
                if (++count == 42)
                {
                    throw vogon_exception();
                }
                return i;
            });

        HPX_TEST(data == dest);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST(hpx::init(argc, argv) == 0);
    return hpx::util::report_errors();
}
