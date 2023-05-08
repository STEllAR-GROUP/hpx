//  Copyright (c) 2013 Mario Mulansky
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #798: HPX_LIMIT does not
// work for local dataflow

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/unwrap.hpp>

// define large action
double func(double x1, double, double, double, double, double, double)
{
    return x1;
}

int hpx_main()
{
    hpx::shared_future<double> f = hpx::make_ready_future(1.0);

    f = hpx::dataflow(
        hpx::launch::sync, hpx::unwrapping(&func), f, f, f, f, f, f, f);
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
#endif
