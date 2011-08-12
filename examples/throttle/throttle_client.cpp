//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <sstream>

#include <hpx/hpx_init.hpp>

#include "throttle/throttle.hpp"

#include <boost/algorithm/string/classification.hpp> 
#include <boost/algorithm/string/split.hpp>
#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using boost::algorithm::is_space;
using boost::algorithm::split;

int hpx_main(variables_map& vm)
{
    {
        std::cout << ( boost::format("prefix: %d")
                     % hpx::naming::get_prefix_from_id(hpx::find_here()))
                  << std::endl;
    }

    hpx::disconnect();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    std::cout << "commands: init [iterations], help, quit\n";

    while (true)
    {
        std::cout << "> ";

        std::string line;
        std::getline(std::cin, line);

        if (line.empty())
            continue;

        else if (0 == std::string("quit").find(line))
            break; 

        std::vector<std::string> call;
        split(call, line, is_space());

        if (0 == std::string("init").find(call[0]))
        {
            try
            {
                boost::uint64_t iterations = 1;

                if (call.size() == 2)
                    iterations = boost::lexical_cast<boost::uint64_t>(call[1]);

                else if (call.size() > 2)
                    std::cout << ( boost::format(
                                   "error: '%s' has too many arguments\n")
                                 % line);
                    
                // Initialize and run HPX.
                for (boost::uint64_t i = 0; i < iterations; ++i)
                {
                    hpx::init(cmdline, argc, argv, hpx::runtime_mode_connect);
                }
            }

            catch (boost::bad_lexical_cast&)
            {
                std::cout << ( boost::format(
                               "error: '%s' is not an unsigned integer\n")
                             % call[1]);
            }

            continue;
        } 

        else if (0 != std::string("help").find(line))
            std::cout << ( boost::format(
                           "error: unknown command '%s'\n")
                         % line);
        
        std::cout << "commands: init [iterations], help, quit\n";
    }
}

