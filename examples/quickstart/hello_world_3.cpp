//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to initialize the _local_ HPX runtime
// explicitly and execute a HPX-thread printing "Hello World!" once. That's all.

//[hello_world_3_getting_started
#include <hpx/init.hpp>

#include <iostream>

int hpx_main(int, char*[])
{
    // Say hello to the world!
    std::cout << "Hello World!\n" << std::flush;
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::local::init(&hpx_main, argc, argv);
}
//]
