//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to disable HPX command
// line alias handling and to allow for unknown options to be passed through
// to hpx_main.

#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>

#include <string>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    for (int i = 0; i != argc; ++i)
        hpx::cout << "arg(" << i << "): " << argv[i] << hpx::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.commandline.aliasing=0",       // disable aliasing
        "hpx.commandline.allow_unknown=1"   // allow for unknown options
    };

    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
