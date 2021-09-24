//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/algorithm.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/numeric.hpp>
#include <hpx/local/tuple.hpp>
#include <hpx/modules/iterator_support.hpp>

#include <iostream>
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

    using hpx::get;
    using hpx::tuple;
    using hpx::util::make_zip_iterator;

    double result = hpx::transform_reduce(hpx::execution::par,
        make_zip_iterator(std::begin(xvalues), std::begin(yvalues)),
        make_zip_iterator(std::end(xvalues), std::end(yvalues)), 0.0,
        std::plus<double>(),
        [](tuple<double, double> r) { return get<0>(r) * get<1>(r); });
    // print the result
    std::cout << result << std::endl;

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
