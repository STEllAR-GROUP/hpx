//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2002-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#include <hpx/hpx_main.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdlib>

static unsigned func_impl(int arg1, bool arg2, double arg3)
{
    using namespace std;
    return abs(static_cast<int>((arg2 ? arg1 : 2 * arg1) * arg3));
}

int main(int, char*[])
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::util::function_nonser<unsigned(bool, double)> f1 =
        hpx::util::bind(func_impl, 15, _1, _2);
    hpx::util::function_nonser<unsigned(double)> f2 =
        hpx::util::bind(f1, false, _1);
    hpx::util::function_nonser<unsigned()> f3 = hpx::util::bind(f2, 4.0);

    HPX_TEST_EQ(f3(), 120);

    return hpx::util::report_errors();
}

