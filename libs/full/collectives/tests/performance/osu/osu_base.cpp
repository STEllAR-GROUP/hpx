//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Bidirectional network bandwidth test

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>

#include <cstddef>

void print_header();
void run_benchmark(hpx::program_options::variables_map& vm);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    print_header();
    run_benchmark(vm);
    return hpx::finalize();
}

#include <iostream>

int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::cerr << "main started...\n";

    // clang-format off
    desc.add_options()
        ("window-size",
         hpx::program_options::value<std::size_t>()->default_value(1),
         "Number of messages to send in parallel")
        ("loop",
         hpx::program_options::value<std::size_t>()->default_value(100),
         "Number of loops")
        ("min-size",
         hpx::program_options::value<std::size_t>()->default_value(1),
         "Minimum size of message to send")
        ("max-size",
         hpx::program_options::value<std::size_t>()->default_value((1<<22)),
         "Maximum size of message to send");
    // clang-format ON

hpx::init_params init_args;
init_args.desc_cmdline = desc;

    return hpx::init(argc, argv, init_args);
}
#endif
