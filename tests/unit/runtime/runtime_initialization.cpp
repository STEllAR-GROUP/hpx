//  Copyright (c) 2018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple test verifying runtime initialization sequence

#define _GNU_SOURCE

#include <dlfcn.h>
#include <iostream>
#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

// For testing hpx::future and hpx::async
int test_future()
{
    return 42;
}


// This will try to use HPX functionality directly from main
int main(int argc, char** argv)
{
    hpx::future<int> f = hpx::async(&test_future);
    HPX_TEST(f.get() == 42);

    return 0;
}
