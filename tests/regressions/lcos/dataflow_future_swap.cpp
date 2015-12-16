//  Copyright (c) 2013 Mario Mulansky
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #775: runtime error with
// local dataflow (copying futures?).

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/format.hpp>
#include <boost/thread/thread.hpp>

#include <iostream>

using hpx::util::unwrapped;

typedef hpx::lcos::shared_future< double > future_type;

struct mul
{
    typedef double result_type;

    double operator()( double x1 , double x2 ) const
    {
        hpx::this_thread::sleep_for( boost::chrono::milliseconds(10000) );
        hpx::cout << boost::format( "func: %f , %f\n" ) % x1 %x2 << hpx::flush;
        return x1*x2;
    }
};

double dummy(double x, double) { std::cout << "dummy: " << x << "\n"; return x; }

void future_swap( future_type &f1 , future_type &f2 )
{
    future_type tmp = f1;
    f1 = hpx::dataflow( unwrapped( &dummy ) , f2 , f1 );
    f2 = hpx::dataflow( unwrapped( &dummy ) , tmp, f1 );
}

int main()
{
    future_type f1 = hpx::make_ready_future( 2.0 );
    future_type f2 = hpx::make_ready_future( 3.0 );

    f1 = hpx::dataflow( unwrapped(mul()) , f1 , f2 );

    future_swap( f1 , f2 );

    hpx::cout << boost::format("f1: %d\n") % f1.get() << hpx::flush;
    hpx::cout << boost::format("f2: %d\n") % f2.get() << hpx::flush;

    return 0;
}
