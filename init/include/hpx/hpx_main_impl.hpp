//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Default implementation of main() if all the user provides is
// hpx_startup::user_main.
//
// This has to be in a separate translation unit to ensure the linker can pick
// or ignore this function, depending on whether the main executable defines
// this symbol or not.
//
// This also enables to pass through any unknown options to make the behavior
// of main() as similar as possible with a real main entry point.
//
// Note: this function is intentionally not marked as inline
//
int HPX_CDECL main(int argc, char** argv)
{
    // allow for unknown options and inhibit aliasing of short options
    std::vector<std::string> const cfg = {
        "hpx.commandline.allow_unknown=1",
        "hpx.commandline.aliasing=0"
    };

    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}

// Make sure header testing code does not redefine main()
#define HPX_MAIN_DEFINED 1

