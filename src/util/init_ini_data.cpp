//  Copyright (c) 2005-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/config/defaults.hpp>
#include <hpx/util/filesystem_compatibility.hpp>
#include <hpx/util/init_ini_data.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/plugins/plugin_registry_base.hpp>

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/tokenizer.hpp>
#include <hpx/util/plugin.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/range/iterator_range.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file (section& ini, std::string const& loc)
    {
        try {
            namespace fs = boost::filesystem;
            boost::system::error_code ec;
            if (!fs::exists(loc, ec) || ec)
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
    bool init_ini_data_base (section& ini, std::string& hpx_ini_file)
    {
        namespace fs = boost::filesystem;

        // fall back: use compile time prefix
        std::string ini_paths(ini.get_entry("hpx.master_ini_path"));
        std::string ini_paths_suffixes(ini.get_entry("hpx.master_ini_path_suffixes"));

        // split off the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer_type;

        boost::char_separator<char> sep (HPX_INI_PATH_DELIMITER);
        tokenizer_type tok_paths(ini_paths, sep);
        tokenizer_type::iterator end_paths = tok_paths.end();
        tokenizer_type tok_suffixes(ini_paths_suffixes, sep);
        tokenizer_type::iterator end_suffixes = tok_suffixes.end();

        bool result = false;
        for (tokenizer_type::iterator it = tok_paths.begin (); it != end_paths; ++it) {
            std::string path = *it;
            for (tokenizer_type::iterator jt = tok_suffixes.begin ();
                jt != end_suffixes; ++jt) {
                path += *jt;
                bool result2 = handle_ini_file (ini, path + "/hpx.ini");
                if (result2) {
                    LBT_(info) << "loaded configuration: " << path << "/hpx.ini";
                }
                result = result2 || result;
            }
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
#if !defined(HPX_WINDOWS)   // /etc/hpx.ini doesn't make sense for Windows
        {
            bool result2 = handle_ini_file(ini, "/etc/hpx.ini");
            if (result2) {
                LBT_(info) << "loaded configuration: " << "/etc/hpx.ini";
            }
            result = result2 || result;
        }
#endif

        result = handle_ini_file_env(ini, "HOME", "/.hpx.ini") || result;
        result = handle_ini_file_env(ini, "PWD", "/.hpx.ini") || result;

        if (!hpx_ini_file.empty()) {
            namespace fs = boost::filesystem;
            boost::system::error_code ec;
            if (!fs::exists(hpx_ini_file, ec) || ec) {
                std::cerr << "hpx::init: command line warning: file specified using "
                             "--hpx::config does not exist ("
                    << hpx_ini_file << ")." << std::endl;
                hpx_ini_file.clear();
                result = false;
            }
            else {
                bool result2 = handle_ini_file(ini, hpx_ini_file);
                if (result2) {
                    LBT_(info) << "loaded configuration: " << hpx_ini_file;
                }
                return result || result2;
            }
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

        // split off the separate paths from the given path list
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

                boost::system::error_code ec;
                if (!fs::exists(this_path, ec) || ec)
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
    void load_component_factory_static(util::section& ini, std::string name,
        hpx::util::plugin::get_plugins_list_type get_factory, error_code& ec)
    {
        hpx::util::plugin::static_plugin_factory<
            components::component_registry_base> pf(get_factory);

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
            // demangle the name in debug mode
            if (name[name.size()-1] == 'd')
                name.resize(name.size()-1);
#endif
            ini_data += std::string("[hpx.components.") + name + "]";
            ini_data += "name = " + name;
            ini_data += "no_factory = 1";
            ini_data += "enabled = 1";
            ini_data += "static = 1";
        }
        else {
            // ask all registries
            for (std::string const& s : names)
            {
                // create the component registry object
                boost::shared_ptr<components::component_registry_base>
                    registry (pf.create(s, ec));
                if (ec) continue;

                registry->get_component_info(ini_data, "", true);
            }
        }

        // incorporate all information from this module's
        // registry into our internal ini object
        ini.parse("<component registry>", ini_data, false, false);
    }

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
            // demangle the name in debug mode
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
            for (std::string const& s : names)
            {
                // create the component registry object
                boost::shared_ptr<components::component_registry_base>
                    registry (pf.create(s, ec));
                if (ec) return;

                registry->get_component_info(ini_data, curr);
            }
        }

        // incorporate all information from this module's
        // registry into our internal ini object
        ini.parse("<component registry>", ini_data, false, false);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
    load_plugin_factory(hpx::util::plugin::dll& d, util::section& ini,
        std::string const& curr, std::string const& name, error_code& ec)
    {
        typedef std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
            plugin_list_type;

        plugin_list_type plugin_registries;
        hpx::util::plugin::plugin_factory<plugins::plugin_registry_base>
            pf(d, "plugin");

        // retrieve the names of all known registries
        std::vector<std::string> names;
        pf.get_names(names, ec);      // throws on error
        if (ec) return plugin_registries;

        std::vector<std::string> ini_data;
        if (!names.empty()) {
            // ask all registries
            for (std::string const& s : names)
            {
                // create the plugin registry object
                boost::shared_ptr<plugins::plugin_registry_base>
                    registry(pf.create(s, ec));
                if (ec) continue;

                registry->get_plugin_info(ini_data);
                plugin_registries.push_back(registry);
            }
        }

        // incorporate all information from this module's
        // registry into our internal ini object
        ini.parse("<plugin registry>", ini_data, false, false);
        return plugin_registries;
    }

    namespace detail
    {
        inline bool cmppath_less(
            std::pair<boost::filesystem::path, std::string> const& lhs,
            std::pair<boost::filesystem::path, std::string> const& rhs)
        {
            return lhs.first < rhs.first;
        }

        inline bool cmppath_equal(
            std::pair<boost::filesystem::path, std::string> const& lhs,
            std::pair<boost::filesystem::path, std::string> const& rhs)
        {
            return lhs.first == rhs.first;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
    init_ini_data_default(std::string const& libs, util::section& ini,
        std::map<std::string, boost::filesystem::path>& basenames,
        std::map<std::string, hpx::util::plugin::dll>& modules)
    {
        namespace fs = boost::filesystem;

        typedef std::vector<std::pair<fs::path, std::string> >::iterator
            iterator_type;

        typedef std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
            plugin_list_type;

        plugin_list_type plugin_registries;

        // list of modules to load
        std::vector<std::pair<fs::path, std::string> > libdata;
        try {
            fs::directory_iterator nodir;
            fs::path libs_path (hpx::util::create_path(libs));

            boost::system::error_code ec;
            if (!fs::exists(libs_path, ec) || ec)
                return plugin_registries;     // given directory doesn't exist

            // retrieve/create section [hpx.components]
            if (!ini.has_section("hpx.components")) {
                util::section* hpx_sec = ini.get_section("hpx");
                HPX_ASSERT(NULL != hpx_sec);

                util::section comp_sec;
                hpx_sec->add_section("components", comp_sec);
            }

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

#if !defined(HPX_WINDOWS)
                if (0 == name.find("lib"))
                    name = name.substr(3);
#endif
#if defined(__APPLE__) // shared library version is added berfore extension
                const std::string version = hpx::full_version_as_string();
                std::string::size_type i = name.find(version);
                if (i != std::string::npos)
                    name.erase(i - 1, version.length() + 1); // - 1 for one more dot
#endif
                // ensure base directory, remove symlinks, etc.
                boost::system::error_code fsec;
                fs::path canonical_curr = util::canonical_path(curr, fsec);
                if (fsec)
                    canonical_curr = curr;

                // make sure every module name is loaded exactly once, the
                // first occurrence of a module name is used
                std::string basename = canonical_curr.filename().string();
                std::pair<std::map<std::string, fs::path>::iterator, bool> p =
                    basenames.insert(std::make_pair(basename, canonical_curr));

                if (p.second) {
                    libdata.push_back(std::make_pair(canonical_curr, name));
                }
                else {
                    LRT_(warning) << "skipping module " << basename
                        << " (" << canonical_curr.string() << ")"
                        << ": ignored because of: " << (*p.first).second.string();
                }
            }
        }
        catch (fs::filesystem_error const& e) {
            LRT_(info) << "caught filesystem error: " << e.what();
        }

        // return if no new modules have been found
        if (libdata.empty())
            return plugin_registries;

        // make sure each node loads libraries in a different order
        std::srand(static_cast<unsigned>(std::time(0)));
        std::random_shuffle(libdata.begin(), libdata.end());

        typedef std::pair<fs::path, std::string> libdata_type;
        for (libdata_type const& p : libdata)
        {
            // get the handle of the library
            error_code ec(lightweight);
            hpx::util::plugin::dll d(p.first.string(), p.second);
            d.load_library(ec);
            if (ec) {
                LRT_(info)
                    << "skipping (load_library failed): " << p.first.string()
                    << ": " << get_error_what(ec);
                continue;
            }

            // get the component factory
            std::string curr_fullname(p.first.parent_path().string());
            load_component_factory(d, ini, curr_fullname, p.second, ec);
            if (ec) {
                LRT_(info)
                    << "skipping (load_component_factory failed): "
                    << p.first.string()
                    << ": " << get_error_what(ec);
                ec = error_code(lightweight);   // reinit ec
            }

            // get the plugin factory
            plugin_list_type tmp_regs =
                load_plugin_factory(d, ini, curr_fullname, p.second, ec);

            std::copy(tmp_regs.begin(), tmp_regs.end(),
                std::back_inserter(plugin_registries));
            if (ec) {
                LRT_(info)
                    << "skipping (load_plugin_factory failed): "
                    << p.first.string()
                    << ": " << get_error_what(ec);
            }

            // store loaded library for future use
            modules.insert(std::make_pair(p.second, std::move(d)));
        }
        return plugin_registries;
    }
}}
