//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

int main()
{
    const int size = 1000000;
    float* a = new float[size];

    bool caught_exception = false;
    try {
        // this should throw as the HPX runtime has not been initialized
        hpx::parallel::fill(hpx::parallel::par, a, a + size, 1.0f);

        // fill should have thrown
        HPX_TEST(false);
    }
    catch (hpx::exception const&) {
        caught_exception = true;
    }

    HPX_TEST(caught_exception);

    delete[] a;

    return 0;
}
