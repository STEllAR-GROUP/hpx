//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test for compatibility with Boost.ProgramOptions
// hpxinspect:nodeprecatedinclude:boost/program_options.hpp

// We use boost::program_options here, so suppress the deprecation warning.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <hpx/hpx_init.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int hpx_main(po::variables_map&)
{
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("compression", po::value<double>(), "set compression level")
    ;
    // clang-format on

    hpx::program_options::options_description cmdline(desc);
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
