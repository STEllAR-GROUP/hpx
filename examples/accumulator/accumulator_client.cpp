//  Copyright (c) 2007-2012 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "accumulator/accumulator.hpp"

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    // get list of all known remote localities supporting our accumulator type
    hpx::components::component_type t =
        hpx::components::accumulator::get_component_type();
    std::vector<hpx::naming::id_type> localities =
        hpx::find_remote_localities(t);

    hpx::naming::id_type prefix;

    if (!localities.empty())
        // create accumulator on any of the remote localities
        prefix = localities[0];
    else
        // create an accumulator locally
        prefix = hpx::find_here();

    {
        // create an accumulator locally
        hpx::components::accumulator accu;
        accu.create(prefix);

        // print some message
        std::cout << "accumulator client, you may enter some commands "
                     "(try 'help' if in doubt...)" << std::endl;

        // execute a couple of commands on this component
        std::string cmd;
        std::cin >> cmd;
        while (std::cin.good())
        {
            if (cmd == "init")
                accu.init();

            else if (cmd == "add")
            {
                std::string arg;
                std::cin >> arg;
                accu.add(boost::lexical_cast<unsigned long>(arg));
            }

            else if (cmd == "print")
                accu.print();

            else if (cmd == "query")
                std::cout << accu.get_gid() << "> " << accu.query() << std::endl;

            else if (cmd == "help")
                std::cout << "commands: init, add [amount], print, query, help, quit"
                          << std::endl;

            else if (cmd == "quit")
                break;

            else
                std::cout << "Invalid command.\n"
                             "commands: init, add [amount], print, help, quit"
                          << std::endl;

            std::cin >> cmd;
        }
    }

    // initiate shutdown of the runtime systems on all localities
    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

