//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #565

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

int invoked_init = 0;

int hpx_main(int argc, char ** argv)
{
    ++invoked_init;
    return hpx::finalize();
}

int main()
{
    // Everything is fine on the first call
    hpx::init();
    HPX_TEST(invoked_init == 1);

    // Segfault on the call, now fixed
    hpx::init();
    HPX_TEST(invoked_init == 2);

    return hpx::util::report_errors();
}
