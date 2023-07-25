//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/chrono.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>

int hpx_main()
{
    hpx::execution::experimental::num_cores nc(2);
    auto policy = hpx::execution::par.with(nc);

    HPX_TEST_EQ(
        hpx::parallel::execution::processing_units_count(policy.parameters(),
            policy.executor(), hpx::chrono::null_duration, 0),
        static_cast<std::size_t>(2));

    auto policy2 =
        hpx::parallel::execution::with_processing_units_count(policy, 2);
    HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                    hpx::execution::par.parameters(), policy2.executor(),
                    hpx::chrono::null_duration, 0),
        static_cast<std::size_t>(2));

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
