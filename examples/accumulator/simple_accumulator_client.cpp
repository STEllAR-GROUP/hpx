//  Copyright (c) 2007-2010 Hartmut Kaiser, Richard D Guidry Jr
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "simple_accumulator/simple_accumulator.hpp"

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map&)
{
    // get list of all known localities
    std::vector<hpx::naming::id_type> prefixes;
    hpx::applier::applier& appl = hpx::applier::get_applier();
    hpx::naming::id_type prefix;

    if (appl.get_remote_prefixes(prefixes))
        // create accumulator on any of the remote localities
        prefix = prefixes[0];
    else
        // create an accumulator locally
        prefix = appl.get_runtime_support_gid();

    hpx::components::simple_accumulator accu;
    accu.create(prefix);

    // print some message
    std::cout << "simple accumulator client, you may enter some commands\n"
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
            accu.add(boost::lexical_cast<double>(arg));
        }

        else if (cmd == "print")
            accu.print();

        else if (cmd == "query")
            std::cout << accu.query() << std::endl;

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

    // free the accumulator component
    accu.free(); // this invalidates the remote reference

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

