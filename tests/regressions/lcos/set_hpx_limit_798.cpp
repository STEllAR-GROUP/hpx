//  Copyright (c) 2013 Mario Mulansky
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #798: HPX_LIMIT does not
// work for local dataflow

// anything > 10 triggers #798
#define HPX_LIMIT 15

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>

using hpx::lcos::shared_future;
using hpx::lcos::make_ready_future;
using hpx::util::unwrapped;

// define large action
double func(double x1 , double x2 , double x3 , double x4 ,
    double x5 , double x6 , double x7)
{
    return x1;
}

int main()
{
    shared_future< double > f = make_ready_future( 1.0 );
    f = hpx::dataflow(
            hpx::launch::sync,
            unwrapped(&func),
            f, f, f, f, f, f, f);
    return hpx::util::report_errors();
}
