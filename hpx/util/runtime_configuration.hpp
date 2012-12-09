//  Copyright (c) 2007-2012 Hartmut Kaiser
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
    class HPX_EXPORT runtime_configuration : public section
    {
        std::string hpx_ini_file;
        std::vector<std::string> cmdline_ini_defs;

    public:
        // initialize and load configuration information
        runtime_configuration();

        // re-initialize all entries based on the additional information from
        // the given configuration file
        void reconfigure(std::string const& ini_file);

        // re-initialize all entries based on the additional information from
        // any explicit command line options
        void reconfigure(std::vector<std::string> const& ini_defs);

        void load_components();

        // Returns the AGAS mode of this locality, returns either hosted (for
        // localities connecting to a remote AGAS server) or bootstrap for the
        // locality hosting the AGAS server.
        agas::service_mode get_agas_service_mode() const;

        // AGAS server only: get number of localities served
        std::size_t get_num_localities() const;

        std::size_t get_agas_promise_pool_size() const;

        // Get the AGAS locality to use
        naming::locality get_agas_locality() const;

        // Get the HPX network address to use
        naming::locality get_parcelport_address() const;

        // Get the number of maximum concurrent connections per locality
        std::size_t get_max_connections_per_loc() const;
        std::size_t get_max_connections() const;

        // Get AGAS client-side local cache size
        std::size_t get_agas_local_cache_size() const;
        std::size_t get_agas_local_cache_size_per_thread() const;

        bool get_agas_caching_mode() const;

        bool get_agas_range_caching_mode() const;

        std::size_t get_agas_max_pending_refcnt_requests() const;

        // Get whether the AGAS server is running as a dedicated runtime.
        // This decides whether the AGAS actions are executed with normal
        // priority (if dedicated) or with high priority (non-dedicated)
        bool get_agas_dedicated_server() const;

        // Load application specific configuration and merge it with the
        // default configuration loaded from hpx.ini
        bool load_application_configuration(char const* filename,
            error_code& ec = throws);

        // Can be set to true if we want to use the ITT notify tools API.
        bool get_itt_notify_mode() const;

        // Returns the number of OS threads this locality is running.
        std::size_t get_os_thread_count() const;

        // Returns the command line that this locality was invoked with.
        std::string get_cmd_line() const;

        // Will return the default stack size to use for all HPX-threads.
        std::ptrdiff_t get_default_stack_size() const
        {
            return small_stacksize;
        }

        // Will return the requested stack size to use for an HPX-threads.
        std::ptrdiff_t get_stack_size(threads::thread_stacksize stacksize) const;

        // Return the configured sizes of any of the know thread pools
        std::size_t get_thread_pool_size(char const* poolname) const;

        // Return the endianess to be used for out-serialization
        std::string get_endian_out() const;

    private:
        std::ptrdiff_t init_stack_size(char const* entryname, 
            char const* defaultvaluestr, std::ptrdiff_t defaultvalue) const;

        std::ptrdiff_t init_small_stack_size() const;
        std::ptrdiff_t init_medium_stack_size() const;
        std::ptrdiff_t init_large_stack_size() const;
        std::ptrdiff_t init_huge_stack_size() const;

#if defined(__linux) || defined(linux) || defined(__linux__)
        void init_use_stack_guard_pages() const;
#endif

        void pre_initialize_ini();
        void post_initialize_ini(std::string const& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs);

    private:
        std::ptrdiff_t small_stacksize;
        std::ptrdiff_t medium_stacksize;
        std::ptrdiff_t large_stacksize;
        std::ptrdiff_t huge_stacksize;
        bool need_to_call_pre_initialize;
    };
}}

#endif
