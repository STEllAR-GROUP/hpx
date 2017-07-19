//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/iostreams.hpp>

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

    using hpx::util::make_zip_iterator;
    using hpx::util::tuple;
    using hpx::util::get;

    double result =
        hpx::parallel::transform_reduce(
            hpx::parallel::execution::par,
            make_zip_iterator(std::begin(xvalues), std::begin(yvalues)),
            make_zip_iterator(std::end(xvalues), std::end(yvalues)),
            0.0,
            std::plus<double>(),
            [](tuple<double, double> r)
            {
                return get<0>(r) * get<1>(r);
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
    return hpx::init(argc, argv, cfg);
}


