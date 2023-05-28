//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

int hpx_main(hpx::program_options::variables_map& vm)
{
    HPX_TEST_EQ(vm.count("integer_option"), static_cast<std::size_t>(1));
    HPX_TEST_EQ(vm["integer_option"].as<int>(), 42);

    HPX_TEST_EQ(vm.count("string_option"), static_cast<std::size_t>(1));
    HPX_TEST_EQ(vm["string_option"].as<std::string>(), std::string("42"));

    HPX_TEST_EQ(vm.count("float_option"), static_cast<std::size_t>(1));
    HPX_TEST_EQ(vm["float_option"].as<double>(), 42.0);

    HPX_TEST_EQ(vm.count("bool_option"), static_cast<std::size_t>(1));

    HPX_TEST_EQ(vm.count("array_option"), static_cast<std::size_t>(1));
    HPX_TEST((vm["array_option"].as<std::vector<int>>() ==
        std::vector<int>{1, 2, 3, 4, 5}));

    HPX_TEST_EQ(vm.count("base:option"), static_cast<std::size_t>(1));
    HPX_TEST_EQ(vm["base:option"].as<int>(), 42);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("integer_option", value<int>(), "test integer_option")
        ("string_option", value<std::string>(), "test string_option")
        ("float_option", value<double>(), "test float_option")
        ("bool_option", "test bool_option")
        ("array_option", value<std::vector<int>>()->composing(),
            "test array_option")
        ("base:option", value<int>(), "test base:option")
    ;
    // clang-format on

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
