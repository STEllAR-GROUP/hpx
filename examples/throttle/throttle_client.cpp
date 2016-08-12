//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include "throttle/throttle.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/format.hpp>

#include <sstream>
#include <string>
#include <vector>

using boost::program_options::variables_map;

using boost::algorithm::is_space;
using boost::algorithm::split;

using hpx::naming::get_agas_client;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    try {
        std::cout << ( boost::format("prefix: %d")
                     % hpx::naming::get_locality_id_from_id(hpx::find_here()))
                  << std::endl;

        // Try to connect to existing throttle instance, create a new one if
        // this fails.
        char const* throttle_component_name = "/throttle/0";
        hpx::naming::id_type gid =
            hpx::agas::resolve_name(hpx::launch::sync, throttle_component_name);
        throttle::throttle t;
        if (!t.get_id()) {
            std::vector<hpx::naming::id_type> localities =
                hpx::find_remote_localities();

            // create throttle on the console, register the instance with AGAS
            // and add an additional reference count to keep it alive
            if (!localities.empty()) {
                // use AGAS client to get the component type as we do not
                // register any factories
                t.create(localities[0]);
                hpx::agas::register_name(hpx::launch::sync,
                    throttle_component_name, t.get_id());
            }
            else {
                std::cerr << "Can't find throttle component." << std::endl;
            }
        }

        // handle commands
        if (t.get_id()) {
            if (vm.count("suspend")) {
                t.suspend(vm["suspend"].as<int>());
            }
            else if (vm.count("resume")) {
                t.resume(vm["resume"].as<int>());
            }
            else if (vm.count("release")) {
                // unregister from AGAS, remove additional reference count which
                // will allow for the throttle instance to be released
                hpx::agas::unregister_name(hpx::launch::sync,
                    throttle_component_name);
            }
        }
    }
    catch (hpx::exception const& e) {
        std::cerr << "throttle_client: caught exception: " << e.what() << std::endl;
    }

    hpx::disconnect();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    // Configure application-specific options
    po::options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ("suspend", po::value<int>(), "suspend thread with given number")
        ("resume", po::value<int>(), "resume thread with given number")
        ("release", "release throttle component instance")
    ;

    // Disable loading of all external components
    std::vector<std::string> const cfg = {
        "hpx.components.load_external=0",
        "hpx.run_hpx_main!=1"
    };

    hpx::util::function_nonser<void()> const empty;
    return hpx::init(cmdline, argc, argv, cfg, empty, empty, hpx::runtime_mode_connect);
}

