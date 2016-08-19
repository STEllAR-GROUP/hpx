//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

#include <boost/format.hpp>

#include <string>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

using hpx::naming::get_agas_client;

using hpx::naming::address;
using hpx::naming::id_type;
using hpx::naming::resolver_client;
using hpx::naming::get_locality_id_from_gid;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        std::cout << "commands: localities, help, quit\n";

        while (true)
        {
            std::cout << "> ";

            std::string arg;
            std::getline(std::cin, arg);

            if (arg.empty())
                continue;

            else if (0 == std::string("quit").find(arg))
                break;

            else if (0 == std::string("localities").find(arg))
            {
                std::vector<id_type> localities = hpx::find_all_localities();

                for (id_type const& locality_ : localities)
                {
                    address addr = hpx::agas::resolve(locality_).get();

                    std::cout << ( boost::format("  [%1%] %2%\n")
                                 % get_locality_id_from_gid(locality_.get_gid())
                                 % addr.locality_);
                }

                continue;
            }

            else if (0 != std::string("help").find(arg))
                std::cout << ( boost::format("error: unknown command '%1%'\n")
                             % arg);

            std::cout << "commands: localities, help, quit\n";
        }
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this application to use at least 2 threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=2"
    };

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv, cfg);
}

