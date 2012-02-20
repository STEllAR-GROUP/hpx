//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>

#include <boost/foreach.hpp>
#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

using hpx::naming::get_agas_client;

using hpx::naming::address;
using hpx::naming::gid_type;
using hpx::naming::resolver_client;
using hpx::naming::get_locality_id_from_gid;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        resolver_client& agas_client = get_agas_client();

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
                std::vector<gid_type> localities;
                agas_client.get_locality_ids(localities);

                BOOST_FOREACH(gid_type const& locality_id, localities)
                {
                    address addr;
                    agas_client.resolve(locality_id, addr);

                    std::cout << ( boost::format("  [%1%] %2%\n")
                                 % get_locality_id_from_gid(locality_id)
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

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

