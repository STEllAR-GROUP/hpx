//  Copyright (c) 2007-2012 Hartmut Kaiser,
//  Copyright (c) 2007 Richard D Guidry Jr.
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>

#include "accumulators/managed_accumulator.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/classification.hpp>

char const* const help = "commands: reset, add [amount], query, help, quit";

void ignore_whitespace(std::istream& is)
{
    while (' ' == is.peek() || '\t' == is.peek())
        is.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    {
        // Create an accumulator component on this locality.
        examples::managed_accumulator accu;
        accu.create(hpx::find_here());

        // Print out the available commands.
        std::cout << help << std::endl << "> ";

        std::cin >> std::noskipws;

        std::string line;

        // Enter the interpreter loop.
        while (std::getline(std::cin, line).good())
        {
            std::vector<std::string> cmd;

            boost::algorithm::trim(line);

            boost::algorithm::split(cmd, line,
                boost::algorithm::is_any_of(" \t\n"),
                boost::algorithm::token_compress_on);

            if (!cmd.empty() && !cmd[0].empty())
            {
                if (cmd[0] == "reset")
                    accu.reset_sync();

                if (cmd[0] == "reset")
                    accu.reset_sync();

                else if (cmd[0] == "add")
                {
                    if (cmd.size() != 2)
                        std::cout << "error: invalid command '"
                                  << line << "'" << std::endl
                                  << help << std::endl;
                    else
                        accu.add_sync(boost::lexical_cast<boost::uint64_t>
                            (cmd[1]));
                }

                else if (cmd[0] == "query")
                    std::cout << accu.query_sync() << std::endl;

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
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

