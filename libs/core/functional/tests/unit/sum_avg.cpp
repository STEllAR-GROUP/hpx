//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org/

#include <hpx/functional/function.hpp>
#include <hpx/modules/testing.hpp>

void do_sum_avg(int values[], int n, int& sum, double& avg)
{
    sum = 0;
    for (int i = 0; i < n; i++)
        sum += values[i];
    avg = (double) sum / n;
}

int main()
{
    hpx::util::function_nonser<void(int values[], int n, int& sum, double& avg)>
        sum_avg;
    sum_avg = &do_sum_avg;

    return hpx::util::report_errors();
}
