//  Copyright (c) 2017 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

std::string expected;

int hpx_init_test(std::string s, int argc, char ** argv)
{
    HPX_TEST_EQ(s, expected);
    return hpx::finalize();
}

int main (int argc, char *argv[])
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    expected = "first";
    hpx::util::function_nonser<int(int, char**)> callback1 =
        hpx::util::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback1, argc, argv);

    expected = "second";
    hpx::util::function_nonser<int(int, char**)> callback2 =
        hpx::util::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback2, argc, argv);

    expected = "third";
    hpx::util::function_nonser<int(int, char**)> callback3 =
        hpx::util::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback3, argc, argv);

    return 0;
}
