//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2008 Richard D Guidry Jr.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include "accumulator.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <string>

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
        examples::accumulator accu(
            hpx::components::new_<accumulator_type>(localities.back()));

        // Print out the available commands.
        std::cout << help << std::endl << "> ";

        // Enter the interpreter loop.
        std::string line;
        while (std::getline(std::cin, line))
        {
            boost::algorithm::trim(line);

            std::vector<std::string> cmd;
            boost::algorithm::split(cmd, line,
                boost::algorithm::is_any_of(" \t\n"),
                boost::algorithm::token_compress_on);

            if (!cmd.empty() && !cmd[0].empty())
            {
                // try to interpret the entered command
                if (cmd[0] == "reset") {
                    accu.reset_sync();
                }
                else if (cmd[0] == "add") {
                    if (cmd.size() == 2) {
                        accu.add_sync(boost::lexical_cast<argument_type>(cmd[1]));
                    }
                    else {
                        std::cout << "error: invalid command '"
                                  << line << "'" << std::endl
                                  << help << std::endl;
                    }
                }
                else if (cmd[0] == "query") {
                    std::cout << accu.query_sync() << std::endl;
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=2");

    // Initialize and run HPX.
    return hpx::init(argc, argv, cfg);
}

