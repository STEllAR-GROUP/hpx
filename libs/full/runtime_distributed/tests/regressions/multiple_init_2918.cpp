//  Copyright (c) 2017 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/testing.hpp>

#include <string>

std::string expected;

int hpx_init_test(std::string s, int, char**)
{
    HPX_TEST_EQ(s, expected);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    expected = "first";
    hpx::function<int(int, char**)> callback1 =
        hpx::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback1, argc, argv);

    expected = "second";
    hpx::function<int(int, char**)> callback2 =
        hpx::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback2, argc, argv);

    expected = "third";
    hpx::function<int(int, char**)> callback3 =
        hpx::bind(&hpx_init_test, expected, _1, _2);
    hpx::init(callback3, argc, argv);

    return 0;
}
