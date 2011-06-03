//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM)
#define HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM

#include <vector>
#include <string>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/ini.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
#if HPX_AGAS_VERSION > 0x10
    void post_initialize_ini(section& ini, std::string const& hpx_ini_file = "",
        std::vector<std::string> const& cmdline_ini_defs = std::vector<std::string>()) HPX_EXPORT;
#endif

    // The runtime_configuration class is a wrapper for the runtime 
    // configuration data allowing to extract configuration information in a 
    // more convenient way
    class HPX_API_EXPORT runtime_configuration : public section
    {
    public:
        // initialize and load configuration information
        runtime_configuration();
        runtime_configuration(std::vector<std::string> const& prefill,
                              std::string const& hpx_ini_file = "",
                              std::vector<std::string> const& cmdline_ini_defs
                                  = std::vector<std::string>());


#if HPX_AGAS_VERSION > 0x10
        agas::router_mode get_agas_router_mode() const;
#endif

        // Get the AGAS locality to use 
        naming::locality get_agas_locality() const;

        // Get the AGAS locality to use (default_address/default_port are 
        // the default values describing the locality to use if no 
        // configuration info can be found).
        naming::locality get_agas_locality(naming::locality const& l) const;

        // Get AGAS client-side GVA cache size
        std::size_t get_agas_gva_cache_size() const;
        
        // Get AGAS client-side locality cache size
        std::size_t get_agas_locality_cache_size() const;

        // Get AGAS connection cache size
        std::size_t get_agas_connection_cache_size() const;

        // Load application specific configuration and merge it with the
        // default configuration loaded from hpx.ini
        bool load_application_configuration(char const* filename, 
            error_code& ec = throws);

        // Can be set to true if we are only going to run HPX in one locality.
        bool get_agas_smp_mode() const;

        // Can be set to true if we want to use the ITT notify tools API 
        bool get_itt_notify_mode() const;
    };

}}

#endif
