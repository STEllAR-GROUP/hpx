//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::variables_map;

using hpx::future;

///////////////////////////////////////////////////////////////////////////////
bool null_thread_executed = false;

bool null_thread()
{
    HPX_TEST(!null_thread_executed);
    null_thread_executed = true;
    return true;
}

// Define the boilerplate code necessary for the function 'null_thread'
// to be invoked as an HPX action (by a HPX future)
HPX_PLAIN_ACTION(null_thread, null_action)

///////////////////////////////////////////////////////////////////////////////
int int_thread()
{
    return 9000;
}

// Define the boilerplate code necessary for the function 'int_thread'
// to be invoked as an HPX action (by a HPX future)
HPX_PLAIN_ACTION(int_thread, int_action)

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        using hpx::async;

        // create an explicit future
        null_thread_executed = false;
        {
            HPX_TEST(async<null_action>(hpx::find_here()).get());
        }
        HPX_TEST(null_thread_executed);

        // create an implicit future
        null_thread_executed = false;
        {
            HPX_TEST(async<null_action>(hpx::find_here()).get());
        }
        HPX_TEST(null_thread_executed);

        //test two successive 'get' from a promise
        hpx::shared_future<int> int_promise(
            async<int_action>(hpx::find_here()));
        HPX_TEST_EQ(int_promise.get(), int_promise.get());
    }

    {
        using hpx::async;
        null_action do_null;

        // create an explicit future
        null_thread_executed = false;
        {
            HPX_TEST(async(do_null, hpx::find_here()).get());
        }
        HPX_TEST(null_thread_executed);

        // create an implicit future
        null_thread_executed = false;
        {
            HPX_TEST(async(do_null, hpx::find_here()).get());
        }
        HPX_TEST(null_thread_executed);

        //test two successive 'get' from a promise
        int_action do_int;
        hpx::shared_future<int> int_promise(async(do_int, hpx::find_here()));
        HPX_TEST_EQ(int_promise.get(), int_promise.get());
    }

    hpx::finalize();    // Initiate shutdown of the runtime system.
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
