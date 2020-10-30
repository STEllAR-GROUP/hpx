//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

hpx::future<void> bcast_void(double)
{
    return hpx::make_ready_future();
}

HPX_PLAIN_ACTION(bcast_void);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(bcast_void_action)
HPX_REGISTER_BROADCAST_ACTION(bcast_void_action)

hpx::future<double> bcast(double bcast)
{
    return hpx::make_ready_future(bcast);
}

HPX_PLAIN_ACTION(bcast);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(bcast_action)
HPX_REGISTER_BROADCAST_ACTION(bcast_action)

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    auto f1 = hpx::lcos::broadcast<bcast_void_action>(localities, 42.0);
    f1.get();

    auto f2 = hpx::lcos::broadcast<bcast_action>(localities, 42.0);
    for (hpx::future<double>& f : f2.get())
    {
        HPX_TEST_EQ(42.0, f.get());
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
