//  Copyright (c) 2013 Mario Mulansky
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #773: local dataflow with
// unwrapped: functor operators need to be const.

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>

typedef hpx::lcos::shared_future< double > future_type;

template< typename Value >
struct mul
{
    const Value a;

    mul( const Value alpha )
      : a( alpha )
    {}

    double operator() ( double x1 , double x2 ) //const // this has to be const?!
    {
        return x1*x2*a;
    }
};

int main()
{
    auto functor = hpx::util::unwrapped(mul<double>( 0.5 ));
    future_type f1 = hpx::make_ready_future( 1.0 );
    future_type f2 = hpx::dataflow( functor , f1 , f1 );

    hpx::wait_all(f1, f2);

    return 0;
}
