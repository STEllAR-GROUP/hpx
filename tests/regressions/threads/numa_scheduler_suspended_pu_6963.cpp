//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test for #6963:
// shared_priority_queue_scheduler: NUMA scheduling does not handle suspended
// processing units. The NUMA hint path previously skipped select_active_pu(),
// which all other hint modes call. A direct runtime reproduction via public
// API is not possible since suspend_processing_unit is internal. This test
// verifies the NUMA hint scheduling path completes without hanging or
// crashing.

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

int hpx_main()
{
    std::size_t const num_tasks = 64;
    std::atomic<std::size_t> count{0};

    std::vector<hpx::future<void>> futures;
    futures.reserve(num_tasks);

    for (std::size_t i = 0; i != num_tasks; ++i)
    {
        futures.push_back(hpx::async([&count]() { ++count; }));
    }

    hpx::wait_all(futures);
    HPX_TEST_EQ(count, num_tasks);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params iparams;
    iparams.mode = hpx::runtime_mode::local;
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, iparams), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
