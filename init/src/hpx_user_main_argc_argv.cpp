//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/config_entry.hpp>

#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
// Forwarding of hpx_startup::user_main, if necessary. This has to be in a
// separate translation unit to ensure the linker can pick or ignore this
// function, depending on whether the main executable defines this symbol
// or not.
int hpx_startup::user_main(int /* argc */, char** /* argv */)
{
    // If we have seen unknown command line options we can throw here as we
    // know that the user is not going to look at the arguments.
    std::string unknown_command_line =
        hpx::get_config_entry("hpx.unknown_cmd_line_option", "");

    if (!unknown_command_line.empty())
    {
        std::cerr << "hpx_startup::user_main: command line processing: "
                     "unknown command line option(s): '"
                  << unknown_command_line << "'\n";
        return -1;
    }

    return hpx_startup::user_main();
}
