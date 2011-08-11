//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include "throttle/throttle.hpp"

#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map& vm)
{
    {
        std::cout << "commands: help, quit\n";

        while (true)
        {
            std::cout << "> ";

            std::string arg;
            std::getline(std::cin, arg);

            if (arg.empty())
                continue;
            
            if (0 == std::string("quit").find(arg))
                break; 

            if (0 != std::string("help").find(arg))
                std::cout << ( boost::format("error: unknown command '%1%'\n")
                             % arg);
            
            std::cout << "commands: help, quit\n";
        }
    }

    hpx::disconnect();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv, hpx::runtime_mode_probe);
}

