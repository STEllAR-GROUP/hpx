//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/get_config_entry.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
// Forwarding of hpx_startup::user_main, if necessary. This has to be in a
// separate translation unit to ensure the linker can pick or ignore this
// function, depending on whether the main executable defines this symbol
// or not.
int hpx_startup::user_main(int argc, char** argv)
{
    // If we have seen unknown command line options we can throw here as we
    // know that the user is not going to look at the arguments.
    std::string unknown_command_line =
        hpx::get_config_entry("hpx.unknown_cmd_line_option", "");

    if (!unknown_command_line.empty())
    {
        hpx::detail::report_exception_and_terminate(
            HPX_GET_EXCEPTION(hpx::bad_parameter, "hpx_startup::user_main",
                "unknown command line option(s): " + unknown_command_line)
        );
    }

    return hpx_startup::user_main();
}
