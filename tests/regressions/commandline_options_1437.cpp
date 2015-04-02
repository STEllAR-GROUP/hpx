//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1437: hpx::init() should strip HPX-related flags from argv

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

bool invoked_main = false;

int my_hpx_main(int argc, char **argv)
{
    // all HPX command line arguments should have been stripped here
    HPX_TEST(argc == 1);

    invoked_main = true;
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST(argc > 1);

    HPX_TEST_EQ(hpx::init(&my_hpx_main, "testapp", argc, argv), 0);
    HPX_TEST(invoked_main);

    return hpx::util::report_errors();
}
