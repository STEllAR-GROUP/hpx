//  Copyright (c) 2005-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if!defined(HPX_INIT_INI_DATA_SEP_26_2008_0344PM)
#define HPX_INIT_INI_DATA_SEP_26_2008_0344PM

#include <string>
#include <iostream>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/tokenizer.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    ///////////////////////////////////////////////////////////////////////////
    inline bool handle_ini_file (section& ini, std::string const& loc)
    {
        try { 
            ini.read (loc); 
        }
        catch (hpx::exception const& /*e*/) { 
            return false;
        }
        return true;
    }

    inline bool handle_ini_file_env (section& ini, char const* env_var, 
        char const* file_suffix = NULL)
    {
        char const* env = getenv(env_var);
        if (NULL != env) {
            namespace fs = boost::filesystem;

            fs::path inipath (env, fs::native);
            if (NULL != file_suffix)
                inipath /= fs::path(file_suffix, fs::native);

            return handle_ini_file(ini, inipath.string());
        }
        return false;
    }


    ///////////////////////////////////////////////////////////////////////////
    // read system and user specified ini files
    //
    // returns true if at least one alternative location has been read 
    // successfully
    inline bool init_ini_data_base (section& ini)
    {
        namespace fs = boost::filesystem;

        // fall back: use compile time prefix
        bool result = handle_ini_file (ini, std::string(HPX_DEFAULT_INI_PATH) + "/hpx.ini");

        // look in the current directory first
        std::string cwd = fs::current_path().string() + "/.hpx.ini";
        result = handle_ini_file (ini, cwd) || result;

        // look for master ini in the HPX_INI environment
        result = handle_ini_file_env (ini, "HPX_INI") || result;

        // afterwards in the standard locations
#if !defined(BOOST_WINDOWS)   // /etc/hpx.ini doesn't make sense for Windows
        result = handle_ini_file(ini, "/etc/hpx.ini") || result;
#endif
        result = handle_ini_file_env(ini, "HPX_LOCATION", "/share/hpx/hpx.ini") || result;
        result = handle_ini_file_env(ini, "HOME", "/.hpx.ini") || result;
        return handle_ini_file_env(ini, "PWD", "/.hpx.ini") || result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // global function to read component ini information
    inline void merge_component_inis(section& ini)
    {
        namespace fs = boost::filesystem;

        // now merge all information into one global structure
        std::string ini_path(HPX_DEFAULT_INI_PATH);
        std::vector <std::string> ini_paths;

        if (ini.has_entry("hpx.ini_path"))
            ini_path = ini.get_entry("hpx.ini_path");

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
                fs::path this_path (*it, fs::native);

                if (!fs::exists(this_path)) 
                    continue;

                for (fs::directory_iterator dir(this_path); dir != nodir; ++dir)
                {
                    if (fs::extension(*dir) != ".ini") 
                        continue;

                    // read and merge the ini file into the main ini hierarchy
                    try {
                        ini.merge ((*dir).string ());
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

///////////////////////////////////////////////////////////////////////////////
}}  

#endif 

