//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

struct test
{
    test() { ++count; }
    test(test const & t) { ++count; }
    test& operator=(test const & t) { ++count; return *this; }
    ~test() { --count; }

    static int count;
};

int test::count = 0;

int hpx_main(hpx::program_options::variables_map & vm)
{
    {
        HPX_TEST_EQ(test::count, 0);
        hpx::lcos::promise<test> p;
        hpx::lcos::future<test> f = p.get_future();
        p.set_value(test());
        HPX_TEST_EQ(test::count, 1);
        f.get();
    }
    // Flush pending reference counting operations.
    hpx::agas::garbage_collect();
    hpx::this_thread::yield();
    hpx::agas::garbage_collect();
    HPX_TEST_EQ(test::count, 0);

    hpx::finalize();

    return hpx::util::report_errors();
}

int main(int argc, char **argv)
{
    hpx::program_options::options_description desc(
        "usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc, argc, argv);
}
