//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/numeric.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/iterator/counting_iterator.hpp>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // lets say we have two vectors that simulate.. 10007D
    std::vector<double> xvalues(10007);
    std::vector<double> yvalues(10007);
    std::fill(boost::begin(xvalues), boost::end(xvalues), 1.0);
    std::fill(boost::begin(yvalues), boost::end(yvalues), 1.0);

    using hpx::util::make_zip_iterator;
    using hpx::util::tuple;
    using hpx::util::make_tuple;
    using hpx::util::get;

    // the result of the execution will be stored in location 0 of the tuple
    double result =
        hpx::parallel::reduce(
            hpx::parallel::par,
            boost::counting_iterator<int>(0),
            boost::counting_iterator<int>(10007),
            0.0,
            std::plus<double>(),
            [&xvalues, &yvalues](__int64 i){
                    return xvalues[i] * yvalues[i];
            }
        );
    // print the result
    hpx::cout << result << hpx::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}
