//  Taken from the Boost.Function library

// Copyright (C) 2001-2003 Douglas Gregor
//  Copyright 2013 Hartmut Kaiser
//
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org/

#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

float mul_ints(int x, int y) { return ((float)x) * y; }

struct int_div
{
    float operator()(int x, int y) const { return ((float)x)/y; };
};

int main()
{
    hpx::util::function_nonser<float(int x, int y)> f;

    f = int_div();
    HPX_TEST(f);
    HPX_TEST_EQ(f(5, 3), 5.f/3);

    f = nullptr;
    HPX_TEST(!f);

    f = &mul_ints;
    HPX_TEST(f);
    HPX_TEST_EQ(f(5, 3), 5.f*3);

    return hpx::util::report_errors();
}
