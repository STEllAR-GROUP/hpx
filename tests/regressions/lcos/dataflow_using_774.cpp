//  Copyright (c) 2013 Mario Mulansky
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #774: using local dataflow
// without explicit namespace.

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>

#include <iostream>

// the following line causes compile errors
using hpx::lcos::local::dataflow;

typedef hpx::lcos::shared_future< double > future_type;

template< typename Value >
struct mul
{
    const Value a;

    mul( const Value alpha )
      : a( alpha )
    {}

    double operator() ( double x1 , double x2 ) const // this has to be const?!
    {
        return x1*x2*a;
    }
};

int main()
{
    auto functor = hpx::util::unwrapped(mul<double>( 0.5 ));
    future_type f1 = hpx::make_ready_future( 1.0 );

    // compile error even when using full namespace
    future_type f2 = dataflow( functor , f1 , f1 );
    future_type f3 = hpx::lcos::local::dataflow(
        hpx::util::unwrapped(mul<double>( 2.0 )) , f1 , f1 );

    hpx::wait_all(f1, f2, f3);

    return 0;
}
