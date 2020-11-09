//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2008 Richard D Guidry Jr.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:noinclude:hpx::util::from_string

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>

#include "accumulator.hpp"

#include <iostream>
#include <string>
#include <vector>

char const* const help = "commands: reset, add [amount], query, help, quit";

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        typedef examples::server::accumulator accumulator_type;
        typedef accumulator_type::argument_type argument_type;

        // Find the localities connected to this application.
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        // Create an accumulator component either on this locality (if the
        // example is executed on one locality only) or on any of the remote
        // localities (otherwise).
        examples::accumulator accu =
            hpx::new_<accumulator_type>(localities.back());

        // Print out the available commands.
        std::cout << help << std::endl << "> ";

        // Enter the interpreter loop.
        std::string line;
        while (std::getline(std::cin, line))
        {
            hpx::string_util::trim(line);

            std::vector<std::string> cmd;
            hpx::string_util::split(cmd, line,
                hpx::string_util::is_any_of(" \t\n"),
                hpx::string_util::token_compress_mode::on);

            if (!cmd.empty() && !cmd[0].empty())
            {
                // try to interpret the entered command
                if (cmd[0] == "reset") {
                    accu.reset();
                }
                else if (cmd[0] == "add") {
                    if (cmd.size() == 2) {
                        accu.add(hpx::util::from_string<argument_type>(cmd[1]));
                    }
                    else {
                        std::cout << "error: invalid command '"
                                  << line << "'" << std::endl
                                  << help << std::endl;
                    }
                }
                else if (cmd[0] == "query") {
                    std::cout << accu.query() << std::endl;
                }
                else if (cmd[0] == "help") {
                    std::cout << help << std::endl;
                }
                else if (cmd[0] == "quit") {
                    break;
                }
                else {
                    std::cout << "error: invalid command '"
                              << line << "'" << std::endl
                              << help << std::endl;
                }
            }

            std:: cout << "> ";
        }
    }

    // Initiate shutdown of the runtime systems on all localities.
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this example to use 2 threads by default as one of the threads
    // will be sitting most of the time in the kernel waiting for user input.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=2"
    };

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}

#endif
