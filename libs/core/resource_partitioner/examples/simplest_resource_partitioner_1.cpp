//  Copyright (c) 2018 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example only creates a resource partitioner without using it. It is
// intended for inclusion in the documentation.

//[body
#include <hpx/init.hpp>

int hpx_main()
{
    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    // Setup the init parameters
    hpx::local::init_params init_args;
    hpx::local::init(hpx_main, argc, argv, init_args);
}
//body]
