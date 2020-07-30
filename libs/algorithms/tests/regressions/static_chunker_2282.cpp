//  Copyright (c) 2016 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/modules/testing.hpp>

int main()
{
    const int size = 1000000;
    float* a = new float[size];

    bool caught_exception = false;
    try
    {
        // this should throw as the HPX runtime has not been initialized
        hpx::fill(hpx::parallel::execution::par, a, a + size, 1.0f);

        // fill should have thrown
        HPX_TEST(false);
    }
    catch (hpx::exception const&)
    {
        caught_exception = true;
    }

    HPX_TEST(caught_exception);

    delete[] a;

    return 0;
}
