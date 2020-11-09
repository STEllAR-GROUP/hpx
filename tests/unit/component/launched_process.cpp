//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include "components/launch_process_test_server.hpp"

#include <chrono>
#include <string>
#include <vector>

int hpx_main(hpx::program_options::variables_map& vm)
{
    // extract command line arguments
    int exit_code = 0;
    if (vm.count("exit_code") != 0)
        exit_code = vm["exit_code"].as<int>();

    std::string set_message("accessed");
    if (vm.count("set_message") != 0)
        set_message = vm["set_message"].as<std::string>();

    std::string component;
    if (vm.count("component") != 0)
    {
        component = vm["component"].as<std::string>();

        // connect to the component
        hpx::components::client<launch_process::test_server> t;
        t.connect_to(component);

        // set the message
        hpx::future<void> f =
            hpx::async(launch_process_set_message_action(), t, set_message);

        // pretend to do some work
        hpx::this_thread::sleep_for(std::chrono::seconds(1));

        // wait for message to be delivered
        f.get();
    }
    else
    {
        // pretend to do some work
        hpx::this_thread::sleep_for(std::chrono::seconds(1));
    }

    hpx::disconnect();
    return exit_code;
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: launched_process_test [options]");

    desc_commandline.add_options()
        ("exit_code,e", value<int>(), "the value to return to the OS")
        ("component,c", value<std::string>(), "the name of the component")
        ("set_message,s", value<std::string>(), "the message to set in the component")
        ;

    std::vector<std::string> const cfg = {
        // Make sure hpx_main above will be executed even if this is not the
        // console locality.
        "hpx.run_hpx_main!=1",

        // This explicitly enables the component we depend on (it is disabled by
        // default to avoid being loaded outside of this test).
        "hpx.components.launch_process_test_server.enabled!=1",

        // Make sure networking will not be disabled
        "hpx.expect_connecting_localities!=1"
    };

    // Note: this uses runtime_mode::connect to instruct this locality to
    // connect to the existing HPX applications
    hpx::init_params init_args;
    init_args.mode = hpx::runtime_mode::connect;
    init_args.cfg = cfg;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
#endif
