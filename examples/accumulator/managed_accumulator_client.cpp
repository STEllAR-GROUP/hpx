//  Copyright (c) 2007-2011 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include "accumulators/managed_accumulator.hpp"

#include <boost/lexical_cast.hpp>

char const* const help
    = "commands: reset, add [amount], query, help, quit";

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    {
        // Create an accumulator component on this locality.
        accumulators::managed_accumulator accu;
        accu.create(hpx::find_here());

        // Print out the available commands.
        std::cout << help << std::endl << "> ";

        std::string cmd;
        std::cin >> cmd;

        // Enter the interpreter loop.
        while (std::cin.good())
        {
            if (cmd == "reset")
                accu.reset_sync();

            else if (cmd == "add")
            {
                std::string arg;
                std::cin >> arg;
                accu.add_sync(boost::lexical_cast<boost::uint64_t>(arg));
            }

            else if (cmd == "query")
                std::cout << accu.query_sync() << std::endl;

            else if (cmd == "help")
                std::cout << help << std::endl;

            else if (cmd == "quit")
                break;

            else
                std::cout << "error: invalid command" << std::endl
                          << help << std::endl;

            std::cout << "> ";
            std::cin >> cmd;
        }
    }

    // Initiate shutdown of the runtime systems on all localities.
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

