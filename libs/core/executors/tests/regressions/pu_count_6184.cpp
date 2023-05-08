//  Copyright (c) 2023 Marco Diers
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #6184: Wrong processing_units_count of restricted_thread_pool_executor

#include <hpx/execution.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>

int hpx_main()
{
    hpx::parallel::execution::restricted_thread_pool_executor executor{0, 3};
    HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(executor),
        std::size_t(3));
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
