//  Copyright (c) 2007-2015 Hartmut Kaiser,
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:noinclude:hpx::util::from_string

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>

#include "template_function_accumulator.hpp"

#include <iostream>
#include <string>
#include <vector>

char const* const help =
    "commands: reset, addint [amount], addfloat [amount], query, help, quit";

inline void ignore_whitespace(std::istream& is)
{
    while (' ' == is.peek() || '\t' == is.peek())
        is.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        // Create an accumulator component on this locality.
        examples::template_function_accumulator accu =
            hpx::new_<examples::template_function_accumulator>(
                hpx::find_here());

        // Print out the available commands.
        std::cout << help << std::endl << "> ";

        std::cin >> std::noskipws;

        std::string line;

        // Enter the interpreter loop.
        while (std::getline(std::cin, line).good())
        {
            std::vector<std::string> cmd;

            hpx::string_util::trim(line);

            hpx::string_util::split(cmd, line,
                hpx::string_util::is_any_of(" \t\n"),
                hpx::string_util::token_compress_mode::on);

            if (!cmd.empty() && !cmd[0].empty())
            {
                if (cmd[0] == "reset")
                    accu.reset();

                else if (cmd[0] == "addint")
                {
                    if (cmd.size() != 2)
                        std::cout << "error: invalid command '"
                                  << line << "'" << std::endl
                                  << help << std::endl;
                    else
                        accu.add(hpx::util::from_string<int>(cmd[1]));
                }

                else if (cmd[0] == "addfloat")
                {
                    if (cmd.size() != 2)
                        std::cout << "error: invalid command '"
                                  << line << "'" << std::endl
                                  << help << std::endl;
                    else
                        accu.add(hpx::util::from_string<double>(cmd[1]));
                }

                else if (cmd[0] == "query")
                    std::cout << accu.query() << std::endl;

                else if (cmd[0] == "help")
                    std::cout << help << std::endl;

                else if (cmd[0] == "quit")
                    break;

                else
                    std::cout << "error: invalid command '"
                              << line << "'" << std::endl
                              << help << std::endl;
            }

            std::cout << "> ";
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
