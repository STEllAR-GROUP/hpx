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

using boost::algorithm::is_space;
using boost::algorithm::split;

using hpx::naming::get_agas_client;

///////////////////////////////////////////////////////////////////////////////
// AGAS helpers
inline void 
register_name(hpx::naming::id_type const& id, std::string const& name)
{
    get_agas_client().registerid(name, id.get_gid()); 
}

inline void unregister_name(std::string const& name)
{
    get_agas_client().unregisterid(name); 
}

inline hpx::naming::id_type query_name(std::string const& name)
{
    hpx::naming::gid_type gid;
    if (get_agas_client().queryid(name, gid))
      return hpx::naming::id_type(gid, hpx::naming::id_type::unmanaged);

    return hpx::naming::invalid_id;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    try {
        std::cout << ( boost::format("prefix: %d")
                     % hpx::naming::get_prefix_from_id(hpx::find_here()))
                  << std::endl;

        // Try to connect to existing throttle instance, create a new one if 
        // this fails.
        char const* throttle_component_name = "/throttle/0";
        throttle::throttle t(query_name(throttle_component_name));
        if (!t.get_gid()) {
            std::vector<hpx::naming::id_type> prefixes;
            hpx::applier::applier& appl = hpx::applier::get_applier();

            // create throttle on the console, register the instance with AGAS
            // and add an additional reference count to keep it alive
            if (appl.get_remote_prefixes(prefixes)) {
                // use AGAS client to get the component type as we do not
                // register any factories
                hpx::components::component_type type =
                    get_agas_client().get_component_id("throttle_throttle_type");
                std::cout << "throttle component type: " << (int)type << std::endl;

                t.create(prefixes[0], type);
                register_name(t.get_gid(), throttle_component_name);
                get_agas_client().incref(t.get_gid().get_gid());
            }
            else {
                std::cerr << "Can't find throttle component." << std::endl;
            }
        }

        // handle commands
        if (t.get_gid()) {
            if (vm.count("suspend")) {
                t.suspend(vm["suspend"].as<int>());
            }
            else if (vm.count("resume")) {
                t.resume(vm["resume"].as<int>());
            }
            else if (vm.count("release")) {
                // unregister from AGAS, remove additional reference count which 
                // will allow for the throttle instance to be released
                hpx::components::component_type type = t.get_component_type();
                get_agas_client().decref(t.get_gid().get_gid(), type);
                unregister_name(throttle_component_name);
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
    po::options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ("suspend", po::value<int>(), "suspend thread with given number")
        ("resume", po::value<int>(), "resume thread with given number")
        ("release", "release throttle component instance")
    ;

    return hpx::init(cmdline, argc, argv, hpx::runtime_mode_connect);
}

