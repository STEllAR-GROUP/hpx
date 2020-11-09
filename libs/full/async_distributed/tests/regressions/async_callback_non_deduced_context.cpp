//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012      Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=52072

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
int test()
{
    return 42;
}
HPX_PLAIN_ACTION(test, test_action);

///////////////////////////////////////////////////////////////////////////////
int future_callback(hpx::lcos::future<int> p)
{
    HPX_TEST(p.has_value());
    int result = p.get();
    HPX_TEST_EQ(result, 42);
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    using hpx::async;
    using hpx::lcos::future;

    {
        test_action do_test;

        future<int> f = async(do_test, hpx::find_here());
        future<int> p = f.then(
            hpx::util::bind(future_callback, hpx::util::placeholders::_1));

        HPX_TEST_EQ(p.get(), 42);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    hpx::program_options::options_description cmdline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
#endif
