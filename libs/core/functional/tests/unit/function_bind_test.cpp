//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2002-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#include <hpx/functional/bind.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdlib>

static unsigned func_impl(int arg1, bool arg2, double arg3)
{
    using namespace std;
    return abs(static_cast<int>((arg2 ? arg1 : 2 * arg1) * arg3));
}

int main(int, char*[])
{
    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    hpx::function<unsigned(bool, double)> f1 = hpx::bind(func_impl, 15, _1, _2);
    hpx::function<unsigned(double)> f2 = hpx::bind(f1, false, _1);
    hpx::function<unsigned()> f3 = hpx::bind(f2, 4.0);

    HPX_TEST_EQ(f3(), 120u);

    return hpx::util::report_errors();
}
