//  Copyright (c) 2019 Steven R. Brandt
//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_fill.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

int hpx_main()
{
    std::size_t const max_targets =
        (std::min)(hpx::get_num_worker_threads(), std::size_t(10));
    ;
    auto targets = hpx::compute::host::get_local_targets();

    std::size_t nondivisible = 0;
    for (std::size_t num_targets = 1; num_targets < max_targets; ++num_targets)
    {
        for (std::size_t num_elems = 1; num_elems < 10; ++num_elems)
        {
            if (num_elems % num_targets != 0)
            {
                ++nondivisible;
            }

            auto local_targets = targets;
            local_targets.resize((std::min)(targets.size(), num_targets));
            hpx::compute::host::block_executor<> exec(local_targets);

            std::vector<int> v(num_elems, 0);
            // Force there to be as many chunks as elements
            hpx::fill(hpx::parallel::execution::par.on(exec).with(
                          hpx::parallel::execution::static_chunk_size(1)),
                v.begin(), v.end(), 1);

            std::for_each(v.begin(), v.end(), [](int x) { HPX_TEST_EQ(x, 1); });
        }
    }

    // We want at least one of the combinations to have
    // num_elems % num_targets != 0
    HPX_TEST_LT(std::size_t(0), nondivisible);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
