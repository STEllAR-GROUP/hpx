//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include "template_accumulator.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <iostream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
REGISTER_TEMPLATE_ACCUMULATOR(double);
REGISTER_TEMPLATE_ACCUMULATOR(int);

///////////////////////////////////////////////////////////////////////////////
char const* const help = "commands: reset, add [amount], query, help, quit";

template <typename T>
void run_template_accumulator(char const* type)
{
    typedef typename examples::server::template_accumulator<T> accumulator_type;
    typedef typename accumulator_type::argument_type argument_type;

    // Find the localities connected to this application.
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    // Create an accumulator component either on this locality (if the
    // example is executed on one locality only) or on any of the remote
    // localities (otherwise).
    examples::template_accumulator<T> accu(
        hpx::components::new_<accumulator_type>(localities.back()));

    // Print out the available commands.
    std::cout << std::endl
        << "Running accumulator accepting argument type: "
        << type << std::endl;
    std::cout << help << std::endl;

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
                accu.reset();
            }
            else if (cmd[0] == "add") {
                if (cmd.size() == 2) {
                    try {
                        double val = boost::lexical_cast<double>(cmd[1]);
                        accu.add(argument_type(val));
                    }
                    catch (boost::bad_lexical_cast const&) {
                        std::cout << "error: invalid argument for add: '"
                                    << cmd[1] << "'" << std::endl;
                    }
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

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // Dispatch to proper accumulator type
    while (std::cin)
    {
        std::cout << std::endl <<
            "Available accumulator types are: "
            "double (press 1) and int (press 2): ";

        char c = '0';
        std::cin >> c;

        if (c == '1')
            run_template_accumulator<double>("double");
        else if (c == '2')
            run_template_accumulator<int>("int");
        else
            break;
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
    return hpx::init(argc, argv, cfg);
}

