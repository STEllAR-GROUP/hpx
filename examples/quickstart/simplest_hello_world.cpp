//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

///////////////////////////////////////////////////////////////////////////////
// The purpose of this example is to execute a HPX-thread printing
// "Hello world" once. That's all.

int hpx_main(boost::program_options::variables_map&)
{
    // Say Hello to the World!
    hpx::cout << "Hello World!\n";

    // Initiate shutdown of the runtime system
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description desc_commandline(
        "usage: hello_world [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

