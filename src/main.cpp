//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>

#include <vector>
#include <string>

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
int main(int argc, char** argv)
{
    // allow for unknown options
    std::vector<std::string> cfg;
    cfg.push_back("hpx.commandline.allow_unknown=1");

    return hpx::init(argc, argv, cfg);
}

