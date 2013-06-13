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
#include <hpx/plugins/plugin_registry_base.hpp>

#include <string>
#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/tokenizer.hpp>
#include <hpx/util/plugin.hpp>
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

            if (handle_ini_file(ini, inipath.string())) {
                LBT_(info) << "loaded configuration (${" << env_var << "}): "
                           << inipath.string();
                return true;
            }
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
        if (result) {
            LBT_(info) << "loaded configuration: " << ini_path << "/hpx.ini";
        }

        // look in the current directory first
        std::string cwd = fs::current_path().string() + "/.hpx.ini";
        {
            bool result2 = handle_ini_file (ini, cwd);
            if (result2) {
                LBT_(info) << "loaded configuration: " << cwd;
            }
            result = result2 || result;
        }

        // look for master ini in the HPX_INI environment
        result = handle_ini_file_env (ini, "HPX_INI") || result;

        // afterwards in the standard locations
#if !defined(BOOST_WINDOWS)   // /etc/hpx.ini doesn't make sense for Windows
        {
            bool result2 = handle_ini_file(ini, "/etc/hpx.ini");
            if (result2) {
                LBT_(info) << "loaded configuration: " << "/etc/hpx.ini";
            }
            result = result2 || result;
        }
#endif
//      FIXME: is this really redundant?
//         result = handle_ini_file_env(ini, "HPX_LOCATION", "/share/hpx/hpx.ini") || result;

        result = handle_ini_file_env(ini, "HOME", "/.hpx.ini") || result;
        result = handle_ini_file_env(ini, "PWD", "/.hpx.ini") || result;

        if (!hpx_ini_file.empty()) {
            bool result2 = handle_ini_file(ini, hpx_ini_file);
            if (result2) {
                LBT_(info) << "loaded configuration: " << hpx_ini_file;
            }
            return result || result2;
        }
        return result;
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
                        LBT_(info) << "loaded configuration: " << (*dir).path().string();
#else
                        ini.merge ((*dir).string());
                        LBT_(info) << "loaded configuration: " << (*dir).string();
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
    void load_component_factory(hpx::util::plugin::dll& d, util::section& ini,
        std::string const& curr, std::string name, error_code& ec)
    {
        hpx::util::plugin::plugin_factory<components::component_registry_base>
            pf(d, "registry");

        // retrieve the names of all known registries
        std::vector<std::string> names;
        pf.get_names(names, ec);
        if (ec) return;

        std::vector<std::string> ini_data;
        if (names.empty()) {
            // This HPX module does not export any factories, but
            // might export startup/shutdown functions. Create some
            // default configuration data.
            using namespace boost::assign;
#if defined(HPX_DEBUG)
            // unmangle the name in debug mode
            if (name[name.size()-1] == 'd')
                name.resize(name.size()-1);
#endif
            ini_data += std::string("[hpx.components.") + name + "]";
            ini_data += "name = " + name;
            ini_data += "path = " + curr;
            ini_data += "no_factory = 1";
            ini_data += "enabled = 1";
        }
        else {
            // ask all registries
            BOOST_FOREACH(std::string const& s, names)
            {
                // create the component registry object
                boost::shared_ptr<components::component_registry_base>
                    registry (pf.create(s, ec));
                if (ec) return;

                registry->get_component_info(ini_data);
            }
        }

        // incorporate all information from this module's
        // registry into our internal ini object
        ini.parse("component registry", ini_data, false);
    }

    void load_plugin_factory(hpx::util::plugin::dll& d, util::section& ini,
        std::string const& curr, std::string const& name, error_code& ec)
    {
        hpx::util::plugin::plugin_factory<plugins::plugin_registry_base>
            pf(d, "plugin");

        // retrieve the names of all known registries
        std::vector<std::string> names;
        pf.get_names(names, ec);      // throws on error
        if (ec) return;

        std::vector<std::string> ini_data;
        if (!names.empty()) {
            // ask all registries
            BOOST_FOREACH(std::string const& s, names)
            {
                // create the plugin registry object
                boost::shared_ptr<plugins::plugin_registry_base>
                    registry(pf.create(s, ec));
                if (ec) return;

                registry->get_plugin_info(ini_data);
            }
        }

        // incorporate all information from this module's
        // registry into our internal ini object
        ini.parse("plugin registry", ini_data, false);
    }

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
                    error_code ec;
                    hpx::util::plugin::dll d(curr.string(), name);

                    d.load_library(ec);
                    if (!ec) {
                        // get the component factory
                        std::string curr_fullname(curr.parent_path().string());
                        load_component_factory(d, ini, curr_fullname, name, ec);
                    }
                    if (ec) {
                        LRT_(info) << "skipping " << curr.string()
                                   << ": " << get_error_what(ec);
                    }
                }
                catch (std::logic_error const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                }
                catch (std::exception const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                }
                catch (...) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": unexpected exception";
                }

                try {
                    // get the handle of the library
                    error_code ec;
                    hpx::util::plugin::dll d(curr.string(), name);

                    d.load_library(ec);
                    if (!ec) {
                        // get the component factory
                        std::string curr_fullname(curr.parent_path().string());
                        load_plugin_factory(d, ini, curr_fullname, name, ec);
                    }
                    if (ec) {
                        LRT_(info) << "skipping " << curr.string()
                                   << ": " << get_error_what(ec);
                    }
                }
                catch (std::logic_error const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                }
                catch (std::exception const& e) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": " << e.what();
                }
                catch (...) {
                    LRT_(info) << "skipping " << curr.string()
                               << ": unexpected exception";
                }
            }
        }
        catch (fs::filesystem_error const& /*e*/) {
            ;
        }
    }
}}
