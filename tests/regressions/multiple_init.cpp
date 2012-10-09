//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #565

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

int hpx_main(int argc, char ** argv)
{
    return hpx::finalize();
}

int main()
{
    // Everything is fine on the first call
    hpx::init();
    // Segfault on the call
    hpx::init();

    return 0;
}
