//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/init_ini_data.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/filesystem_compatibility.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>

#include <string>
#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/tokenizer.hpp>
#include <boost/plugin.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file (section& ini, std::string const& loc)
    {
        try {
            namespace fs = boost::filesystem;
            if (!fs::exists(loc))
                return false;       // avoid exception on missing file
            ini.read (loc);
        }
        catch (hpx::exception const& /*e*/) {
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file_env (section& ini, char const* env_var,
        char const* file_suffix)
    {
        char const* env = getenv(env_var);
        if (NULL != env) {
            namespace fs = boost::filesystem;

            fs::path inipath (hpx::util::create_path(env));
            if (NULL != file_suffix)
                inipath /= hpx::util::create_path(file_suffix);

            return handle_ini_file(ini, inipath.string());
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // read system and user specified ini files
    //
    // returns true if at least one alternative location has been read
    // successfully
    bool init_ini_data_base (section& ini, std::string const& hpx_ini_file)
    {
        namespace fs = boost::filesystem;

        // fall back: use compile time prefix
        std::string ini_path(ini.get_entry("hpx.master_ini_path"));
        bool result = handle_ini_file (ini, ini_path + "/hpx.ini");

        // look in the current directory first
        std::string cwd = fs::current_path().string() + "/.hpx.ini";
        result = handle_ini_file (ini, cwd) || result;

        // look for master ini in the HPX_INI environment
        result = handle_ini_file_env (ini, "HPX_INI") || result;

        // afterwards in the standard locations
#if !defined(BOOST_WINDOWS)   // /etc/hpx.ini doesn't make sense for Windows
        result = handle_ini_file(ini, "/etc/hpx.ini") || result;
#endif
//      FIXME: is this really redundant?
//         result = handle_ini_file_env(ini, "HPX_LOCATION", "/share/hpx/hpx.ini") || result;

        result = handle_ini_file_env(ini, "HOME", "/.hpx.ini") || result;
        result =  handle_ini_file_env(ini, "PWD", "/.hpx.ini") || result;
        return handle_ini_file(ini, hpx_ini_file) || result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // global function to read component ini information
    void merge_component_inis(section& ini)
    {
        namespace fs = boost::filesystem;

        // now merge all information into one global structure
        std::string ini_path(ini.get_entry("hpx.ini_path", HPX_DEFAULT_INI_PATH));
        std::vector <std::string> ini_paths;

        // split of the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer_type;

        boost::char_separator<char> sep (HPX_INI_PATH_DELIMITER);
        tokenizer_type tok(ini_path, sep);
        tokenizer_type::iterator end = tok.end();
        for (tokenizer_type::iterator it = tok.begin (); it != end; ++it)
            ini_paths.push_back (*it);

        // have all path elements, now find ini files in there...
        std::vector<std::string>::iterator ini_end = ini_paths.end();
        for (std::vector<std::string>::iterator it = ini_paths.begin();
             it != ini_end; ++it)
        {
            try {
                fs::directory_iterator nodir;
                fs::path this_path (hpx::util::create_path(*it));

                if (!fs::exists(this_path))
                    continue;

                for (fs::directory_iterator dir(this_path); dir != nodir; ++dir)
                {
                    if (fs::extension(*dir) != ".ini")
                        continue;

                    // read and merge the ini file into the main ini hierarchy
                    try {
#if BOOST_FILESYSTEM_VERSION == 3
                        ini.merge ((*dir).path().string());
#else
                        ini.merge ((*dir).string());
#endif
                    }
                    catch (hpx::exception const& /*e*/) {
                        ;
                    }
                }
            }
            catch (fs::filesystem_error const& /*e*/) {
                ;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // iterate over all shared libraries in the given directory and construct
    // default ini settings assuming all of those are components
    void init_ini_data_default(std::string const& libs, util::section& ini)
    {
        namespace fs = boost::filesystem;

        try {
            fs::directory_iterator nodir;
            fs::path libs_path (hpx::util::create_path(libs));

            if (!fs::exists(libs_path))
                return;     // given directory doesn't exist

            // retrieve/create section [hpx.components]
            if (!ini.has_section("hpx.components")) {
                util::section* hpx_sec = ini.get_section("hpx");
                BOOST_ASSERT(NULL != hpx_sec);

                util::section comp_sec;
                hpx_sec->add_section("components", comp_sec);
            }

// FIXME: make sure this isn't needed anymore for sure
//             util::section* components_sec = ini.get_section("hpx.components");
//             BOOST_ASSERT(NULL != components_sec);

            // generate component sections for all found shared libraries
            // this will create too many sections, but the non-components will
            // be filtered out during loading
            for (fs::directory_iterator dir(libs_path); dir != nodir; ++dir)
            {
                fs::path curr(*dir);
                if (fs::extension(curr) != HPX_SHARED_LIB_EXTENSION)
                    continue;

                // instance name and module name are the same
                std::string name(fs::basename(curr));

#if !defined(BOOST_WINDOWS)
                if (0 == name.find("lib"))
                    name = name.substr(3);
#endif

                try {
                    // get the handle of the library
                    boost::plugin::dll d(curr.string(), name);

                    // get the factory
                    boost::plugin::plugin_factory<components::component_registry_base>
                        pf(d, "registry");

                    // retrieve the names of all known registries
                    std::vector<std::string> names;
                    pf.get_names(names);      // throws on error

                    std::vector<std::string> ini_data;
                    if (names.empty()) {
                        // This HPX module does not export any factories, but
                        // might export startup/shutdown functions. Create some
                        // default configuration data.
                        using namespace boost::assign;
                        ini_data += std::string("[hpx.components.") + name + "]";
                        ini_data += "name = " + name;
                        ini_data += "path = " + curr.parent_path().string();
                        ini_data += "no_factory = 1";
                        ini_data += "enabled = 1";
                    }
                    else {
                        // ask all registries
                        BOOST_FOREACH(std::string const& s, names)
                        {
                            // create the component registry object
                            boost::shared_ptr<components::component_registry_base>
                                registry (pf.create(s));
                            registry->get_component_info(ini_data);
                        }
                    }

                    // incorporate all information from this module's
                    // registry into our internal ini object
                    ini.parse("component registry", ini_data);

                    continue;   // handle next module
                }
                catch (std::logic_error const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                    continue;   // handle next module
                }
                catch (std::exception const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                    continue;   // handle next module
                }

// FIXME: make sure this isn't needed anymore for sure
            // if something went wrong while reading the registry, just use
            // some default settings
//                 util::section sec;
//                 sec.add_entry("name", name);
//                 sec.add_entry("path", libs);
//                 sec.add_entry("isdefault", "true");
//                 components_sec->add_section(name, sec);
            }
        }
        catch (fs::filesystem_error const& /*e*/) {
            ;
        }
    }
}}
