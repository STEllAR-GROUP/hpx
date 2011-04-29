//  Copyright (c) 2005-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/init_ini_data.hpp>
#include <hpx/util/itt_notify.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    // pre-initialize entries with compile time based values
    void pre_initialize_ini(section& ini)
    {
        using namespace boost::assign;
        std::vector<std::string> lines; 
        lines +=
            // create an empty application section
//            "[application]",

            // create system and application instance specific entries
            "[system]",
            "pid = " + boost::lexical_cast<std::string>(getpid()),
            "prefix = " HPX_PREFIX,

            // create default installation location and logging settings
            "[hpx]",
            "location = ${HPX_LOCATION:$[system.prefix]}",
            "component_path = $[hpx.location]/lib",
            "ini_path = $[hpx.location]/share/hpx/ini",
#if HPX_USE_ITT == 1
            "use_ittnotify = ${HPX_USE_ITTNOTIFY:0}",
#endif
            "finalize_wait_time = ${HPX_FINALIZE_WAIT_TIME:-1.0}",

            "[hpx.agas]",
            "address = ${HPX_AGAS_SERVER_ADRESS:" 
                HPX_NAME_RESOLVER_ADDRESS "}",
            "port = ${HPX_AGAS_SERVER_PORT:" 
                BOOST_PP_STRINGIZE(HPX_NAME_RESOLVER_PORT) "}",
            "cachesize = ${HPX_AGAS_CACHE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_AGAS_CACHE_SIZE) "}",
            "connectioncachesize = ${HPX_AGAS_CONNECTION_CACHE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_MAX_AGAS_CONNECTION_CACHE_SIZE) "}",
            "smp_mode = ${HPX_AGAS_SMP_MODE:0}",

            // create default ini entries for memory_block component hosted in 
            // the main hpx shared library
//             "[hpx.components.memory_block]",
//             "name = hpx",
//             "path = $[hpx.location]/lib/" 
//                 BOOST_PP_STRINGIZE(HPX_MANGLE_NAME(HPX_COMPONENT_NAME))
//                 HPX_SHARED_LIB_EXTENSION,

            // create default ini entries for raw_counter component hosted in 
            // the main hpx shared library
//             "[hpx.components.raw_counter]",
//             "name = hpx",
//             "path = $[hpx.location]/lib/" 
//                 BOOST_PP_STRINGIZE(HPX_MANGLE_NAME(HPX_COMPONENT_NAME))
//                 HPX_SHARED_LIB_EXTENSION,

             "[hpx.components.barrier]",
             "name = hpx",
             "path = $[hpx.location]/lib/" HPX_LIBRARY_STRING,

             "[hpx.components.symbol_namespace]",
             "name = hpx",
             "path = $[hpx.location]/lib/" HPX_LIBRARY_STRING
        ;
        // don't overload user overrides
        ini.parse("static defaults", lines);
    }

    void post_initialize_ini(section& ini,
        std::string const& hpx_ini_file = "",
        std::vector<std::string> const& cmdline_ini_defs
            = std::vector<std::string>())
    {
        // add explicit configuration information if its provided
        util::init_ini_data_base(ini, hpx_ini_file); 

        // let the command line override the config file. 
        if (!cmdline_ini_defs.empty())
            ini.parse("command line definitions", cmdline_ini_defs);

        // try to build default ini structure from shared libraries in default 
        // installation location, this allows to install simple components
        // without the need to install an ini file
        if (ini.has_section("hpx") &&
            ini.get_section("hpx")->has_entry("component_path"))
            util::init_ini_data_default
                (ini.get_section("hpx")->get_entry("component_path"), ini);
        else
            util::init_ini_data_default(HPX_DEFAULT_COMPONENT_PATH, ini);

        // merge all found ini files of all components
        util::merge_component_inis(ini);

        // read system and user ini files _again_, to allow the user to 
        // overwrite the settings from the default component ini's. 
        util::init_ini_data_base(ini, hpx_ini_file);
        
        // let the command line override the config file. 
        if (!cmdline_ini_defs.empty())
            ini.parse("command line definitions", cmdline_ini_defs);
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime_configuration::runtime_configuration()
    {
        pre_initialize_ini(*this);
        post_initialize_ini(*this);
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime_configuration::runtime_configuration(
        std::vector<std::string> const& prefill,
        std::string const& hpx_ini_file,
        std::vector<std::string> const& cmdline_ini_defs)
    {
        pre_initialize_ini(*this);

        if (!prefill.empty())
            this->parse("static prefill defaults", prefill);

        post_initialize_ini(*this, hpx_ini_file, cmdline_ini_defs);

        // set global config options
#if HPX_USE_ITT == 1
        use_ittnotify_api = get_itt_notify_mode();
#endif
    }

    // AGAS configuration information has to be stored in the global hpx.agas
    // configuration section:
    // 
    //    [hpx.agas]
    //    address=<ip address>   # this defaults to HPX_NAME_RESOLVER_ADDRESS
    //    port=<ip port>         # this defaults to HPX_NAME_RESOLVER_PORT
    //
    naming::locality runtime_configuration::get_agas_locality() const
    {
        // load all components as described in the configuration information
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                std::string cfg_port(
                    sec->get_entry("port", HPX_NAME_RESOLVER_PORT));

                return naming::locality(
                    sec->get_entry("address", HPX_NAME_RESOLVER_ADDRESS),
                    boost::lexical_cast<boost::uint16_t>(cfg_port));
            }
        }
        return naming::locality(HPX_NAME_RESOLVER_ADDRESS, HPX_NAME_RESOLVER_PORT);
    }

    naming::locality runtime_configuration::get_agas_locality(
        naming::locality const& l) const
    {
        // load all components as described in the configuration information
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                // read fall back values from configuration file, if needed
                std::string default_address (l.get_address());
                boost::uint16_t default_port = l.get_port();

                if (default_address.empty()) {
                    default_address = 
                        sec->get_entry("address", HPX_NAME_RESOLVER_ADDRESS);
                }
                if (0 == default_port) {
                    default_port = boost::lexical_cast<boost::uint16_t>(
                        sec->get_entry("port", HPX_NAME_RESOLVER_PORT));
                }
                return naming::locality(default_address, default_port);
            }
        }
        return naming::locality(HPX_NAME_RESOLVER_ADDRESS, HPX_NAME_RESOLVER_PORT);
    }

    std::size_t runtime_configuration::get_agas_cache_size() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry("cachesize", HPX_INITIAL_AGAS_CACHE_SIZE));
            }
        }
        return HPX_INITIAL_AGAS_CACHE_SIZE;
    }

    std::size_t runtime_configuration::get_agas_connection_cache_size() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry("connectioncachesize", 
                        HPX_MAX_AGAS_CONNECTION_CACHE_SIZE));
            }
        }
        return HPX_MAX_AGAS_CONNECTION_CACHE_SIZE;
    }

    bool runtime_configuration::get_agas_smp_mode() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("smp_mode", "0")) ? true : false;
            }
        }
        return false;
    }

    bool runtime_configuration::get_itt_notify_mode() const
    {
#if HPX_USE_ITT == 1
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("use_ittnotify", "0")) ? true : false;
            }
        }
#endif
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool runtime_configuration::load_application_configuration(
        char const* filename, error_code& ec)
    {
        try {
            section appcfg(filename);
            section applroot;
            applroot.add_section("application", appcfg);
            this->section::merge(applroot);
        }
        catch (hpx::exception const& e) {
            // file doesn't exist or is ill-formed
            if (&ec == &throws)
                throw;
            ec = make_error_code(e.get_error(), e.what(), hpx::rethrow);
            return false;
        }
        return true;
    }

}}

