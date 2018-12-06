//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example only creates a resource partitioner without using it. It is
// intended for inclusion in the documentation.

//[body
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/resource/partitioner.hpp>

int hpx_main(int argc, char* argv[])
{
    return hpx::finalize();
}

int main(int argc, char** argv)
{
    hpx::resource::partitioner rp(argc, argv);
    hpx::init();
}
//body]
