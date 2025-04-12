//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/experimental/run_on_all.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <cstdint>

int main()
{
    using namespace hpx::experimental;

    {
        std::atomic<std::uint32_t> n(0);
        run_on_all([&]() { ++n; });
        HPX_TEST_EQ(n.load(), hpx::get_num_worker_threads());

        n.store(0);
        run_on_all(2, [&]() { ++n; });
        HPX_TEST_EQ(n.load(), static_cast<std::uint32_t>(2));
    }

    {
        std::uint32_t n = 0;
        run_on_all(
            reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(n, hpx::get_num_worker_threads());

        n = 0;
        run_on_all(
            2, reduction_plus(n), [](std::uint32_t& local_n) { ++local_n; });
        HPX_TEST_EQ(n, static_cast<std::uint32_t>(2));
    }

    return hpx::util::report_errors();
}
