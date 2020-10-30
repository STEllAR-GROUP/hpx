//  Copyright (c) 2013 Mario Mulansky
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #778: swapping futures
// segfaults.

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/async_local/dataflow.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <chrono>
#include <iostream>

using hpx::util::unwrapping;

typedef hpx::lcos::shared_future<double> future_type;

struct mul
{
    double operator()(double x1, double x2) const
    {
        //std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
        hpx::util::format_to(hpx::cout, "func: {}, {}\n", x1, x2) << hpx::flush;
        return x1 * x2;
    }
};

struct divide
{
    double operator()(double x1, double x2) const
    {
        //std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
        hpx::util::format_to(hpx::cout, "func: {}, {}\n", x1, x2) << hpx::flush;
        return x1 / x2;
    }
};

void future_swap(future_type& f1, future_type& f2)
{
    //future_type tmp = hpx::dataflow(
    //  unwrapping( []( double x ){ return x; } ) , f1 );
    future_type tmp = f1;
    f1 = hpx::dataflow(unwrapping([](double x, double) { return x; }), f2, f1);
    f2 = hpx::dataflow(unwrapping([](double x, double) { return x; }), tmp, f1);
}

int main()
{
    future_type f1 = hpx::make_ready_future(2.0);
    future_type f2 = hpx::make_ready_future(3.0);

    for (int n = 0; n < 20; ++n)
    {
        f1 = hpx::dataflow(hpx::launch::async, unwrapping(mul()), f1, f2);
        f2 = hpx::dataflow(hpx::launch::async, unwrapping(divide()), f1, f2);
        future_swap(f1, f2);
    }

    hpx::cout << "futures ready\n" << hpx::flush;

    hpx::util::format_to(hpx::cout, "f1: {}\n", f1.get()) << hpx::flush;
    hpx::util::format_to(hpx::cout, "f2: {}\n", f2.get()) << hpx::flush;

    return 0;
}
#endif
