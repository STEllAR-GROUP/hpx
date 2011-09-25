//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM)
#define HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM

#include <vector>
#include <string>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/util/ini.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // The runtime_configuration class is a wrapper for the runtime 
    // configuration data allowing to extract configuration information in a 
    // more convenient way
    class HPX_API_EXPORT runtime_configuration : public section
    {
    std::string hpx_ini_file;
    std::vector<std::string> cmdline_ini_defs;
 
    public:
        // initialize and load configuration information
        runtime_configuration();
        runtime_configuration(std::vector<std::string> const& prefill,
                              std::string const& hpx_ini_file_ = "",
                              std::vector<std::string> const& cmdline_ini_defs_
                                  = std::vector<std::string>());

        void load_components();

        // Returns the AGAS mode of this locality, returns either hosted (for
        // localities connecting to a remote AGAS server) or bootstrap for the
        // locality hosting the AGAS server.
        agas::router_mode get_agas_router_mode() const;

        // AGAS server only: get number of localities served
        std::size_t get_num_localities() const;

        std::size_t get_agas_allocate_response_pool_size() const;

        std::size_t get_agas_bind_response_pool_size() const;

        // Get the AGAS locality to use 
        naming::locality get_agas_locality() const;

        // Get the AGAS locality to use (default_address/default_port are 
        // the default values describing the locality to use if no 
        // configuration info can be found).
        naming::locality get_agas_locality(naming::locality const& l) const;

        // Get AGAS client-side GVA cache size
        std::size_t get_agas_gva_cache_size() const;
        
        // Get AGAS connection cache size
        std::size_t get_agas_connection_cache_size() const;

        // Load application specific configuration and merge it with the
        // default configuration loaded from hpx.ini
        bool load_application_configuration(char const* filename, 
            error_code& ec = throws);

        // Can be set to true if we want to use the ITT notify tools API.
        bool get_itt_notify_mode() const;

        // Returns the number of OS threads this locality is running.
        std::size_t get_num_os_threads() const;

        // Returns the command line that this locality was invoked with.
        std::string get_cmd_line() const;
    };
}}

#endif
