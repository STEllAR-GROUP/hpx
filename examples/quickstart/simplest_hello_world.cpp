//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a HPX-thread printing
// "Hello World!" once. That's all.

//[simplest_hello_world_getting_started
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

int hpx_main()
{
    // Say hello to the world!
    hpx::cout << "Hello World!\n" << hpx::flush;

    return hpx::finalize();   // Initiate shutdown of the runtime system.
}

int main()
{
    return hpx::init();       // Initialize and run HPX.
}
//]

