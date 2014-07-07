//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/numeric.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
 
#include <hpx/util/lightweight_test.hpp>
 
///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    //lets say we have two vectors that simulate.. 10007D
    std::vector<double> xvalues(10007);
    std::vector<double> yvalues(10007);
    std::fill(boost::begin(xvalues), boost::end(xvalues), 1.0);
    std::fill(boost::begin(yvalues), boost::end(yvalues), 1.0);

    typedef std::vector<double>::iterator iterType;

    //zip the xvalues and yvalues together
    typedef hpx::parallel::util::zip_iterator<iterType, iterType> zip_iterator;
    typedef zip_iterator::reference reference;
    
    //the result of the execution will be stored in location 0 of the tuple
    boost::tuple<double,double> result = hpx::parallel::reduce(hpx::parallel::par, 
        hpx::parallel::util::make_zip_iterator(boost::begin(xvalues), boost::begin(yvalues)),
        hpx::parallel::util::make_zip_iterator(boost::end(xvalues), boost::end(yvalues)),
        boost::make_tuple(0.0,0.0), 
        [](boost::tuple<double,double> res, reference it) {
            return boost::make_tuple(
                boost::get<0>(res) + hpx::util::get<0>(it) * hpx::util::get<1>(it), 
                0.0);
        });

    //print the result
    std::cout << boost::get<0>(result);

    return hpx::finalize();
}
 
int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));
 
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");
 
    return hpx::util::report_errors();
}