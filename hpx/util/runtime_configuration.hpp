//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM)
#define HPX_UTIL_RUNTIME_CONFIGURATION_OCT_02_2008_0530PM

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/runtime/components/static_factory_data.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/plugin/dll.hpp>
#include <hpx/plugins/plugin_registry_base.hpp>

#include <boost/cstdint.hpp>

#include <vector>
#include <string>

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
        runtime_configuration(char const* argv0);

        // re-initialize all entries based on the additional information from
        // the given configuration file
        void reconfigure(std::string const& ini_file);

        // re-initialize all entries based on the additional information from
        // any explicit command line options
        void reconfigure(std::vector<std::string> const& ini_defs);

        std::vector<boost::shared_ptr<plugins::plugin_registry_base> >
            load_modules();

        void load_components_static(std::vector<
            components::static_factory_load_data_type> const& static_modules);

        // Returns the AGAS mode of this locality, returns either hosted (for
        // localities connecting to a remote AGAS server) or bootstrap for the
        // locality hosting the AGAS server.
        agas::service_mode get_agas_service_mode() const;

        // initial number of localities
        boost::uint32_t get_num_localities() const;
        void set_num_localities(boost::uint32_t);

        // sequence number of first usable pu
        boost::uint32_t get_first_used_core() const;
        void set_first_used_core(boost::uint32_t);

        // Get the size of the ipc parcelport data buffer cache
        std::size_t get_ipc_data_buffer_cache_size() const;

        // Get AGAS client-side local cache size
        std::size_t get_agas_local_cache_size(
            std::size_t dflt = HPX_AGAS_LOCAL_CACHE_SIZE) const;

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

        // Enable lock detection during suspension
        bool enable_lock_detection() const;

        // Enable global lock tracking
        bool enable_global_lock_detection() const;

        // Enable minimal deadlock detection for HPX threads
        bool enable_minimal_deadlock_detection() const;

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

        // Return maximally allowed message sizes
        boost::uint64_t get_max_inbound_message_size() const;
        boost::uint64_t get_max_outbound_message_size() const;

        std::map<std::string, hpx::util::plugin::dll> & modules()
        {
            return modules_;
        }

    private:
        std::ptrdiff_t init_stack_size(char const* entryname,
            char const* defaultvaluestr, std::ptrdiff_t defaultvalue) const;

        std::ptrdiff_t init_small_stack_size() const;
        std::ptrdiff_t init_medium_stack_size() const;
        std::ptrdiff_t init_large_stack_size() const;
        std::ptrdiff_t init_huge_stack_size() const;

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        bool init_use_stack_guard_pages() const;
#endif

        void pre_initialize_ini();
        void post_initialize_ini(std::string& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs);

        void reconfigure();

    private:
        mutable boost::uint32_t num_localities;
        std::ptrdiff_t small_stacksize;
        std::ptrdiff_t medium_stacksize;
        std::ptrdiff_t large_stacksize;
        std::ptrdiff_t huge_stacksize;
        bool need_to_call_pre_initialize;
#if defined(__linux) || defined(linux) || defined(__linux__)
        char const* argv0;
#endif

        std::map<std::string, hpx::util::plugin::dll> modules_;
    };
}}

#endif
