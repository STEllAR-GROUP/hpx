//  Copyright (c) 2017 Christopher HInx
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/modules/testing.hpp>

int main()
{
    auto const& partitioner = hpx::resource::get_partitioner();

    bool caught_exception = false;
    try
    {
        // Passing 0 as the number of threads throws an exception.
        hpx::threads::executors::local_priority_queue_os_executor exec(
            0, partitioner.get_affinity_data());
    }
    catch (...)
    {
        caught_exception = true;
    }
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
