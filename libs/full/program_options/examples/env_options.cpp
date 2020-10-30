// Copyright Thomas Kent 2016
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/program_options.hpp>

#include <algorithm>
#include <cctype>
#include <functional>
#include <iostream>
#include <string>

namespace po = hpx::program_options;

std::string mapper(std::string env_var)
{
    // ensure the env_var is all caps
    std::transform(env_var.begin(), env_var.end(), env_var.begin(),
        [](char c) { return std::toupper(c); });

    if (env_var == "PATH")
        return "path";
    if (env_var == "EXAMPLE_VERBOSE")
        return "verbosity";

    return "";
}

void get_env_options()
{
    po::options_description config("Configuration");
    // clang-format off
    config.add_options()
        ("path", "the execution path")
        ("verbosity", po::value<std::string>()->default_value("INFO"),
         "set verbosity: DEBUG, INFO, WARN, ERROR, FATAL")
        ;
    // clang-format on

    po::variables_map vm;
    store(po::parse_environment(
              config, std::function<std::string(std::string)>(mapper)),
        vm);
    notify(vm);

    if (vm.count("path"))
    {
        std::cout << "First 75 chars of the system path: \n";
        std::cout << vm["path"].as<std::string>().substr(0, 75) << std::endl;
    }

    std::cout << "Verbosity: " << vm["verbosity"].as<std::string>()
              << std::endl;
}

int main()
{
    get_env_options();

    return 0;
}
