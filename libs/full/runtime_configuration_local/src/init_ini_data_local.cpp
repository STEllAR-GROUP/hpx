//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/runtime_configuration_local/init_ini_data_local.hpp>
#include <hpx/version.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file(section& ini, std::string const& loc)
    {
        try
        {
            namespace fs = filesystem;
            std::error_code ec;
            if (!fs::exists(loc, ec) || ec)
                return false;    // avoid exception on missing file
            ini.read(loc);
        }
        catch (hpx::exception const& /*e*/)
        {
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file_env(
        section& ini, char const* env_var, char const* file_suffix)
    {
        char const* env = getenv(env_var);
        if (nullptr != env)
        {
            namespace fs = filesystem;

            fs::path inipath(env);
            if (nullptr != file_suffix)
                inipath /= fs::path(file_suffix);

            if (handle_ini_file(ini, inipath.string()))
            {
                LBT_(info) << "loaded configuration (${" << env_var
                           << "}): " << inipath.string();
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
    bool init_ini_data_base(section& ini, std::string& hpx_ini_file)
    {
        namespace fs = filesystem;

        // fall back: use compile time prefix
        std::string ini_paths(ini.get_entry("hpx.master_ini_path"));
        std::string ini_paths_suffixes(
            ini.get_entry("hpx.master_ini_path_suffixes"));

        // split off the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer_type;

        boost::char_separator<char> sep(HPX_INI_PATH_DELIMITER);
        tokenizer_type tok_paths(ini_paths, sep);
        tokenizer_type::iterator end_paths = tok_paths.end();
        tokenizer_type tok_suffixes(ini_paths_suffixes, sep);
        tokenizer_type::iterator end_suffixes = tok_suffixes.end();

        bool result = false;
        for (tokenizer_type::iterator it = tok_paths.begin(); it != end_paths;
             ++it)
        {
            for (tokenizer_type::iterator jt = tok_suffixes.begin();
                 jt != end_suffixes; ++jt)
            {
                std::string path = *it;
                path += *jt;
                bool result2 = handle_ini_file(ini, path + "/hpx.ini");
                if (result2)
                {
                    LBT_(info)
                        << "loaded configuration: " << path << "/hpx.ini";
                }
                result = result2 || result;
            }
        }

        // look in the current directory first
        std::string cwd = fs::current_path().string() + "/.hpx.ini";
        {
            bool result2 = handle_ini_file(ini, cwd);
            if (result2)
            {
                LBT_(info) << "loaded configuration: " << cwd;
            }
            result = result2 || result;
        }

        // look for master ini in the HPX_INI environment
        result = handle_ini_file_env(ini, "HPX_INI") || result;

        // afterwards in the standard locations
#if !defined(HPX_WINDOWS)    // /etc/hpx.ini doesn't make sense for Windows
        {
            bool result2 = handle_ini_file(ini, "/etc/hpx.ini");
            if (result2)
            {
                LBT_(info) << "loaded configuration: "
                           << "/etc/hpx.ini";
            }
            result = result2 || result;
        }
#endif

        result = handle_ini_file_env(ini, "HOME", ".hpx.ini") || result;
        result = handle_ini_file_env(ini, "PWD", ".hpx.ini") || result;

        if (!hpx_ini_file.empty())
        {
            namespace fs = filesystem;
            std::error_code ec;
            if (!fs::exists(hpx_ini_file, ec) || ec)
            {
                std::cerr
                    << "hpx::init: command line warning: file specified using "
                       "--hpx:config does not exist ("
                    << hpx_ini_file << ")." << std::endl;
                hpx_ini_file.clear();
                result = false;
            }
            else
            {
                bool result2 = handle_ini_file(ini, hpx_ini_file);
                if (result2)
                {
                    LBT_(info) << "loaded configuration: " << hpx_ini_file;
                }
                return result || result2;
            }
        }
        return result;
    }
}}    // namespace hpx::util
