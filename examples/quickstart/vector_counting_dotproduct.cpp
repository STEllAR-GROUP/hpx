//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/numeric.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/iterator_support.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // lets say we have two vectors that simulate.. 10007D
    std::vector<double> xvalues(10007);
    std::vector<double> yvalues(10007);
    std::fill(std::begin(xvalues), std::end(xvalues), 1.0);
    std::fill(std::begin(yvalues), std::end(yvalues), 1.0);

    double result =
        hpx::transform_reduce(
            hpx::execution::par,
            hpx::util::counting_iterator<size_t>(0),
            hpx::util::counting_iterator<size_t>(10007),
            0.0,
            std::plus<double>(),
            [&xvalues, &yvalues](size_t i)
            {
                return xvalues[i] * yvalues[i];
            }
        );
    // print the result
    hpx::cout << result << hpx::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
