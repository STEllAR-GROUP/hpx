//  Copyright (c) 2016 Lukas Troska
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This demonstrated the compilation error when using an action that returns a
// future inside dataflow (issue #2008)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

hpx::future<double> foo()
{
    return hpx::make_ready_future(42.);
}

HPX_DEFINE_PLAIN_ACTION(foo);

int hpx_main()
{
    hpx::future<hpx::future<double>> f =
        hpx::dataflow(foo_action(), hpx::find_here());

    HPX_TEST_EQ(f.get().get(), 42.);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
#endif
