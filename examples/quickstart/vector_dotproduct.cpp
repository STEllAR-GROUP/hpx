//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/numeric.hpp>
#include <hpx/include/algorithm.hpp>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // lets say we have two vectors that simulate.. 10007D
    std::vector<double> xvalues(10007);
    std::vector<double> yvalues(10007);
    std::fill(boost::begin(xvalues), boost::end(xvalues), 1.0);
    std::fill(boost::begin(yvalues), boost::end(yvalues), 1.0);

    typedef std::vector<double>::iterator iterator_type;

    using hpx::util::make_zip_iterator;
    using hpx::util::tuple;
    using hpx::util::make_tuple;
    using hpx::util::get;

    // zip the xvalues and yvalues together
    typedef hpx::util::zip_iterator<iterator_type, iterator_type> zip_iterator;
    typedef zip_iterator::reference reference;

    // the result of the execution will be stored in location 0 of the tuple
    double result =
        hpx::parallel::reduce(hpx::parallel::par,
            make_zip_iterator(boost::begin(xvalues), boost::begin(yvalues)),
            make_zip_iterator(boost::end(xvalues), boost::end(yvalues)),
            0.0,
            [](double const& res, double const& g) {
                return res+ g;
            }, [](tuple<double,double> const& r){
                    return get<0>(r) * get<1>(r);           
        });

    // print the result
    std::cout << result;

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
