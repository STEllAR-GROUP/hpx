//  Copyright (c) 2005-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/version.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/init_ini_data.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/find_prefix.hpp>
#include <hpx/util/register_locks.hpp>

#include <boost/config.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/tokenizer.hpp>

#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_alternative.hpp>
#include <boost/spirit/include/qi_sequence.hpp>

///////////////////////////////////////////////////////////////////////////////
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
namespace hpx { namespace util { namespace coroutines { namespace detail { namespace posix
{
    ///////////////////////////////////////////////////////////////////////////
    // this global (urghhh) variable is used to control whether guard pages
    // will be used or not
    HPX_EXPORT bool use_guard_pages = true;
}}}}}
#endif

namespace hpx { namespace threads { namespace policies
{
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
    ///////////////////////////////////////////////////////////////////////////
    // We globally control whether to do minimal deadlock detection using this
    // global bool variable. It will be set once by the runtime configuration
    // startup code
    bool minimal_deadlock_detection = true;
#endif
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    // pre-initialize entries with compile time based values
    void runtime_configuration::pre_initialize_ini()
    {
        if (!need_to_call_pre_initialize)
            return;

        using namespace boost::assign;
        std::vector<std::string> lines;
        lines +=
            // create an empty application section
            "[application]",

            // create system and application instance specific entries
            "[system]",
            "pid = " + boost::lexical_cast<std::string>(getpid()),
            "prefix = " + find_prefix(),
            "executable_prefix = " + get_executable_prefix(),

            // create default installation location and logging settings
            "[hpx]",
            "location = ${HPX_LOCATION:$[system.prefix]}",
            "component_path = $[hpx.location]/lib/hpx"
                HPX_INI_PATH_DELIMITER "$[system.executable_prefix]/lib/hpx"
                HPX_INI_PATH_DELIMITER "$[system.executable_prefix]/../lib/hpx",
            "master_ini_path = $[hpx.location]/share/" HPX_BASE_DIR_NAME
                HPX_INI_PATH_DELIMITER "$[system.executable_prefix]/share/" HPX_BASE_DIR_NAME
                HPX_INI_PATH_DELIMITER "$[system.executable_prefix]/../share/" HPX_BASE_DIR_NAME,
#if HPX_HAVE_ITTNOTIFY != 0
            "use_itt_notify = ${HPX_HAVE_ITTNOTIFY:0}",
#endif
            "finalize_wait_time = ${HPX_FINALIZE_WAIT_TIME:-1.0}",
            "shutdown_timeout = ${HPX_SHUTDOWN_TIMEOUT:-1.0}",
#if HPX_HAVE_VERIFY_LOCKS
            "lock_detection = ${HPX_LOCK_DETECTION:0}",
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
#if HPX_DEBUG
            "minimal_deadlock_detection = ${MINIMAL_DEADLOCK_DETECTION:1}",
#else
            "minimal_deadlock_detection = ${MINIMAL_DEADLOCK_DETECTION:0}",
#endif
#endif

            // add placeholders for keys to be added by command line handling
            "os_threads = 1",
            "localities = 1",
            "runtime_mode = console",

            "[hpx.stacks]",
            "small_size = ${HPX_SMALL_STACK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_SMALL_STACK_SIZE) "}",
            "medium_size = ${HPX_MEDIUM_STACK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_MEDIUM_STACK_SIZE) "}",
            "large_size = ${HPX_LARGE_STACK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_LARGE_STACK_SIZE) "}",
            "huge_size = ${HPX_HUGE_STACK_SIZE:"
                BOOST_PP_STRINGIZE(HPX_HUGE_STACK_SIZE) "}",
#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
            "use_guard_pages = ${HPX_USE_GUARD_PAGES:1}",
#endif

            "[hpx.threadpools]",
            "io_pool_size = ${HPX_NUM_IO_POOL_THREADS:"
                BOOST_PP_STRINGIZE(HPX_NUM_IO_POOL_THREADS) "}",
            "parcel_pool_size = ${HPX_NUM_PARCEL_POOL_THREADS:"
                BOOST_PP_STRINGIZE(HPX_NUM_PARCEL_POOL_THREADS) "}",
            "timer_pool_size = ${HPX_NUM_TIMER_POOL_THREADS:"
                BOOST_PP_STRINGIZE(HPX_NUM_TIMER_POOL_THREADS) "}",

            "[hpx.parcel]",
            "address = ${HPX_PARCEL_SERVER_ADDRESS:" HPX_INITIAL_IP_ADDRESS "}",
            "port = ${HPX_PARCEL_SERVER_PORT:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_IP_PORT) "}",
            "bootstrap = ${HPX_PARCEL_BOOTSTRAP:" HPX_PARCEL_BOOTSTRAP "}",
            "max_connections = ${HPX_PARCEL_MAX_CONNECTIONS:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_CONNECTIONS) "}",
            "max_connections_per_locality = ${HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY) "}",
            "max_message_size = ${HPX_PARCEL_MAX_MESSAGE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_PARCEL_MAX_MESSAGE_SIZE) "}",
#ifdef BOOST_BIG_ENDIAN
            "endian_out = ${HPX_PARCEL_ENDIAN_OUT:big}",
#else
            "endian_out = ${HPX_PARCEL_ENDIAN_OUT:little}",
#endif
            "array_optimization = ${HPX_PARCEL_ARRAY_OPTIMIZATION:1}",

            // TCPIP related settings
            "[hpx.parcel.tcpip]",
            "enable = ${HPX_HAVE_PARCELPORT_TCPIP:1}",

            // shmem related settings
            "[hpx.parcel.shmem]",
            "enable = ${HPX_HAVE_PARCELPORT_SHMEM:0}",
            "data_buffer_cache_size = ${HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE:512}",

            // ibverbs related settings
            "[hpx.parcel.ibverbs]",
            "enable = ${HPX_HAVE_PARCELPORT_IBVERBS:0}",
            "buffer_size = ${HPX_PARCEL_IBVERBS_BUFFER_SIZE:65536}",

            // MPI related settings
            "[hpx.parcel.mpi]",
            "enable = ${HPX_HAVE_PARCELPORT_MPI:0}",

            // predefine command line aliases
            "[hpx.commandline]",
            "-a = --hpx:agas",
            "-c = --hpx:console",
            "-h = --hpx:help",
            "--help = --hpx:help",
            "-I = --hpx:ini",
            "-l = --hpx:localities",
            "-p = --hpx:app-config",
            "-q = --hpx:queuing",
            "-r = --hpx:run-agas-server",
            "-t = --hpx:threads",
            "-v = --hpx:version",
            "--version = --hpx:version",
            "-w = --hpx:worker",
            "-x = --hpx:hpx",
            "-0 = --hpx:node=0",
            "-1 = --hpx:node=1",
            "-2 = --hpx:node=2",
            "-3 = --hpx:node=3",
            "-4 = --hpx:node=4",
            "-5 = --hpx:node=5",
            "-6 = --hpx:node=6",
            "-7 = --hpx:node=7",
            "-8 = --hpx:node=8",
            "-9 = --hpx:node=9",

            "[hpx.agas]",
            "address = ${HPX_AGAS_SERVER_ADDRESS:" HPX_INITIAL_IP_ADDRESS "}",
            "port = ${HPX_AGAS_SERVER_PORT:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_IP_PORT) "}",
            "max_pending_refcnt_requests = "
                "${HPX_AGAS_MAX_PENDING_REFCNT_REQUESTS:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_AGAS_MAX_PENDING_REFCNT_REQUESTS)
                "}",
            "service_mode = hosted",
            "dedicated_server = 0",
            "local_cache_size = ${HPX_AGAS_LOCAL_CACHE_SIZE:"
                BOOST_PP_STRINGIZE(HPX_INITIAL_AGAS_LOCAL_CACHE_SIZE) "}",
            "local_cache_size_per_thread = ${HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD:"
                BOOST_PP_STRINGIZE(HPX_AGAS_LOCAL_CACHE_SIZE_PER_THREAD) "}",
            "use_range_caching = ${HPX_AGAS_USE_RANGE_CACHING:1}",
            "use_caching = ${HPX_AGAS_USE_CACHING:1}",

            "[hpx.components]",
            "load_external = ${HPX_LOAD_EXTERNAL_COMPONENTS:1}",

            "[hpx.components.barrier]",
            "name = hpx",
            "path = $[hpx.location]/lib/hpx/" HPX_DLL_STRING,
            "enabled = 1",

            "[hpx.components.raw_counter]",
            "name = hpx",
            "path = $[hpx.location]/lib/hpx/" HPX_DLL_STRING,
            "enabled = 1",

            "[hpx.components.average_count_counter]",
            "name = hpx",
            "path = $[hpx.location]/lib/hpx/" HPX_DLL_STRING,
            "enabled = 1",

            "[hpx.components.elapsed_time_counter]",
            "name = hpx",
            "path = $[hpx.location]/lib/hpx/" HPX_DLL_STRING,
            "enabled = 1"
        ;

        // don't overload user overrides
        this->parse("static defaults", lines, false);

        need_to_call_pre_initialize = false;
    }

    void runtime_configuration::post_initialize_ini(
        std::string const& hpx_ini_file,
        std::vector<std::string> const& cmdline_ini_defs)
    {
        // add explicit configuration information if its provided
        if (!hpx_ini_file.empty()) {
            util::init_ini_data_base(*this, hpx_ini_file);
            need_to_call_pre_initialize = true;
        }

        // let the command line override the config file.
        if (!cmdline_ini_defs.empty()) {
            this->parse("command line definitions", cmdline_ini_defs);
            need_to_call_pre_initialize = true;
        }
    }

    void runtime_configuration::load_components(
        std::map<std::string, hpx::util::plugin::dll>& modules)
    {
        namespace fs = boost::filesystem;

        // try to build default ini structure from shared libraries in default
        // installation location, this allows to install simple components
        // without the need to install an ini file
        // split of the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char> > tokenizer_type;

        std::string component_path(
            get_entry("hpx.component_path", HPX_DEFAULT_COMPONENT_PATH));

        // protect against duplicate paths
        std::set<std::string> component_paths;

        // list of base names avoiding to load a module more than once
        std::map<std::string, fs::path> basenames;

        boost::char_separator<char> sep (HPX_INI_PATH_DELIMITER);
        tokenizer_type tok(component_path, sep);
        tokenizer_type::iterator end = tok.end();
        for (tokenizer_type::iterator it = tok.begin(); it != end; ++it)
        {
            if (!(*it).empty()) {
                fs::path this_p(*it);
                boost::system::error_code fsec;
                fs::path canonical_p = util::canonical_path(this_p, fsec);
                if (fsec)
                    canonical_p = this_p;

                std::pair<std::set<std::string>::iterator, bool> p =
                    component_paths.insert(
                        util::native_file_string(canonical_p));
                if (p.second) {
                    // have all path elements, now find ini files in there...
                    fs::path this_path (hpx::util::create_path(*p.first));
                    if (fs::exists(this_path)) {
                        util::init_ini_data_default(
                            this_path.string(), *this, basenames, modules);
                    }
                }
            }
        }

        // read system and user ini files _again_, to allow the user to
        // overwrite the settings from the default component ini's.
        util::init_ini_data_base(*this, hpx_ini_file);

        // let the command line override the config file.
        if (!cmdline_ini_defs.empty())
            parse("command line definitions", cmdline_ini_defs);

        // merge all found ini files of all components
        util::merge_component_inis(*this);

        need_to_call_pre_initialize = true;

        // invoke last reconfigure
        reconfigure();
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime_configuration::runtime_configuration()
      : small_stacksize(HPX_SMALL_STACK_SIZE),
        medium_stacksize(HPX_MEDIUM_STACK_SIZE),
        large_stacksize(HPX_LARGE_STACK_SIZE),
        huge_stacksize(HPX_HUGE_STACK_SIZE),
        need_to_call_pre_initialize(true)
    {
        pre_initialize_ini();

        // set global config options
#if HPX_HAVE_ITTNOTIFY != 0
        use_ittnotify_api = get_itt_notify_mode();
#endif
        BOOST_ASSERT(init_small_stack_size() >= HPX_SMALL_STACK_SIZE);

        small_stacksize = init_small_stack_size();
        medium_stacksize = init_medium_stack_size();
        large_stacksize = init_large_stack_size();
        BOOST_ASSERT(init_huge_stack_size() <= HPX_HUGE_STACK_SIZE);
        huge_stacksize = init_huge_stack_size();

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        coroutines::detail::posix::use_guard_pages = init_use_stack_guard_pages();
#endif
#if HPX_HAVE_VERIFY_LOCKS
        if (enable_lock_detection())
            util::enable_lock_detection();
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        threads::policies::minimal_deadlock_detection =
            enable_minimal_deadlock_detection();
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime_configuration::reconfigure(
        std::string const& hpx_ini_file_)
    {
        hpx_ini_file = hpx_ini_file_;
        reconfigure();
    }

    void runtime_configuration::reconfigure(
        std::vector<std::string> const& cmdline_ini_defs_)
    {
        cmdline_ini_defs = cmdline_ini_defs_;
        reconfigure();
    }

    void runtime_configuration::reconfigure()
    {
        pre_initialize_ini();

        std::vector<std::string> const& prefill =
            util::detail::get_logging_data();
        if (!prefill.empty())
            this->parse("static prefill defaults", prefill, false);

        post_initialize_ini(hpx_ini_file, cmdline_ini_defs);

        // set global config options
#if HPX_HAVE_ITTNOTIFY != 0
        use_ittnotify_api = get_itt_notify_mode();
#endif
        BOOST_ASSERT(init_small_stack_size() >= HPX_SMALL_STACK_SIZE);

        small_stacksize = init_small_stack_size();
        medium_stacksize = init_medium_stack_size();
        large_stacksize = init_large_stack_size();
        huge_stacksize = init_huge_stack_size();

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
        coroutines::detail::posix::use_guard_pages = init_use_stack_guard_pages();
#endif
#if HPX_HAVE_VERIFY_LOCKS
        if (enable_lock_detection())
            util::enable_lock_detection();
#endif
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        threads::policies::minimal_deadlock_detection =
            enable_minimal_deadlock_detection();
#endif
    }

    // AGAS configuration information has to be stored in the global hpx.agas
    // configuration section:
    //
    //    [hpx.agas]
    //    address=<ip address>   # this defaults to HPX_INITIAL_IP_PORT
    //    port=<ip port>         # this defaults to HPX_INITIAL_IP_ADDRESS
    //
    // TODO: implement for AGAS v2
    naming::locality runtime_configuration::get_agas_locality() const
    {
        // load all components as described in the configuration information
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                std::string cfg_port(
                    sec->get_entry("port", HPX_INITIAL_IP_PORT));

                return
                    naming::locality(
                        sec->get_entry("address", HPX_INITIAL_IP_ADDRESS)
                      , boost::lexical_cast<boost::uint16_t>(cfg_port)
#if defined(HPX_HAVE_PARCELPORT_MPI)
                      , mpi_environment::enabled() ? 0 : -1
#endif
                    );
            }
        }
        return
            naming::locality(
                HPX_INITIAL_IP_ADDRESS
              , HPX_INITIAL_IP_PORT
#if defined(HPX_HAVE_PARCELPORT_MPI)
              , mpi_environment::enabled() ? 0 : -1
#endif
            );
    }

    // HPX network address configuration information has to be stored in the
    // global hpx configuration section:
    //
    //    [hpx.parcel]
    //    address=<ip address>   # this defaults to HPX_INITIAL_IP_PORT
    //    port=<ip port>         # this defaults to HPX_INITIAL_IP_ADDRESS
    //
    naming::locality runtime_configuration::get_parcelport_address() const
    {
        // load all components as described in the configuration information
        if (has_section("hpx.parcel")) {
            util::section const* sec = get_section("hpx.parcel");
            if (NULL != sec) {
                std::string cfg_port(
                    sec->get_entry("port", HPX_INITIAL_IP_PORT));

                return naming::locality(
                    sec->get_entry("address", HPX_INITIAL_IP_ADDRESS),
                    boost::lexical_cast<boost::uint16_t>(cfg_port));
            }
        }
        return naming::locality(HPX_INITIAL_IP_ADDRESS, HPX_INITIAL_IP_PORT);
    }

    std::size_t runtime_configuration::get_max_connections_per_loc() const
    {
        if (has_section("hpx.parcel"))
        {
            util::section const * sec = get_section("hpx.parcel");
            if(NULL != sec)
            {
                std::string cfg_max_connections_per_loc(
                    sec->get_entry("max_connections_per_locality",
                        HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY));

                return boost::lexical_cast<std::size_t>(cfg_max_connections_per_loc);
            }
        }
        return HPX_PARCEL_MAX_CONNECTIONS_PER_LOCALITY;
    }

    std::size_t runtime_configuration::get_max_connections() const
    {
        if (has_section("hpx.parcel"))
        {
            util::section const * sec = get_section("hpx.parcel");
            if(NULL != sec)
            {
                std::string cfg_max_connections(
                    sec->get_entry("max_connections",
                        HPX_PARCEL_MAX_CONNECTIONS));

                return boost::lexical_cast<std::size_t>(cfg_max_connections);
            }
        }
        return HPX_PARCEL_MAX_CONNECTIONS;
    }

    std::size_t runtime_configuration::get_shmem_data_buffer_cache_size() const
    {
        if (has_section("hpx.parcel"))
        {
            util::section const * sec = get_section("hpx.parcel.shmem");
            if(NULL != sec)
            {
                std::string cfg_shmem_data_buffer_cache_size(
                    sec->get_entry("data_buffer_cache_size",
                        HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE));

                return boost::lexical_cast<std::size_t>(cfg_shmem_data_buffer_cache_size);
            }
        }
        return HPX_PARCEL_SHMEM_DATA_BUFFER_CACHE_SIZE;
    }

    agas::service_mode runtime_configuration::get_agas_service_mode() const
    {
        // load all components as described in the configuration information
        if (has_section("hpx.agas"))
        {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec)
            {
                std::string const m = sec->get_entry("service_mode", "hosted");

                if (m == "hosted")
                    return agas::service_mode_hosted;
                else if (m == "bootstrap")
                    return agas::service_mode_bootstrap;
                else {
                    // REVIEW: exception type is overused
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "runtime_configuration::get_agas_service_mode",
                        std::string("invalid AGAS router mode \"") + m + "\"");
                }
            }
        }
        return agas::service_mode_hosted;
    }

    std::size_t runtime_configuration::get_num_localities() const
    {
        // this function should only be called on the AGAS server
        BOOST_ASSERT(agas::service_mode_bootstrap == get_agas_service_mode());

        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry("localities", 1));
            }
        }
        return 1;
    }

    std::size_t runtime_configuration::get_agas_local_cache_size(std::size_t dflt) const
    {
        std::size_t cache_size = dflt;

        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                cache_size = boost::lexical_cast<std::size_t>(
                    sec->get_entry("local_cache_size", cache_size));
            }
        }

        if (cache_size != std::size_t(~0x0ul) && cache_size < 16ul)
            cache_size = 16;      // limit lower bound
        return cache_size;
    }

    std::size_t runtime_configuration::get_agas_local_cache_size_per_thread(std::size_t dflt) const
    {
        std::size_t cache_size = dflt;

        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                cache_size = boost::lexical_cast<std::size_t>(
                    sec->get_entry("local_cache_size_per_thread", cache_size));
            }
        }

        if (cache_size != std::size_t(~0x0ul) && cache_size < 16ul)
            cache_size = 16;      // limit lower bound
        return cache_size;
    }

    bool runtime_configuration::get_agas_caching_mode() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("use_caching", "1")) ? true : false;
            }
        }
        return false;
    }

    bool runtime_configuration::get_agas_range_caching_mode() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("use_range_caching", "1")) ? true : false;
            }
        }
        return false;
    }

    std::size_t
    runtime_configuration::get_agas_max_pending_refcnt_requests() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry("max_pending_refcnt_requests",
                        HPX_INITIAL_AGAS_MAX_PENDING_REFCNT_REQUESTS));
            }
        }
        return HPX_INITIAL_AGAS_MAX_PENDING_REFCNT_REQUESTS;
    }

    // Get whether the AGAS server is running as a dedicated runtime.
    // This decides whether the AGAS actions are executed with normal
    // priority (if dedicated) or with high priority (non-dedicated)
    bool runtime_configuration::get_agas_dedicated_server() const
    {
        if (has_section("hpx.agas")) {
            util::section const* sec = get_section("hpx.agas");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("dedicated_server", 0)) ? true : false;
            }
        }
        return false;
    }

    bool runtime_configuration::get_itt_notify_mode() const
    {
#if HPX_HAVE_ITTNOTIFY != 0
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("use_itt_notify", "0")) ? true : false;
            }
        }
#endif
        return false;
    }

    // Enable lock detection during suspension
    bool runtime_configuration::enable_lock_detection() const
    {
#if HPX_HAVE_VERIFY_LOCKS
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return boost::lexical_cast<int>(
                    sec->get_entry("lock_detection", "0")) ? true : false;
            }
        }
#endif
        return false;
    }

    // Enable minimal deadlock detection for HPX threads
    bool runtime_configuration::enable_minimal_deadlock_detection() const
    {
#if HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
#if HPX_DEBUG
                return boost::lexical_cast<int>(
                    sec->get_entry("minimal_deadlock_detection", "1")) ? true : false;
#else
                return boost::lexical_cast<int>(
                    sec->get_entry("minimal_deadlock_detection", "0")) ? true : false;
#endif
            }
        }

#if HPX_DEBUG
        return true;
#else
        return false;
#endif

#else
        return false;
#endif
    }

    std::size_t runtime_configuration::get_os_thread_count() const
    {
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry("os_threads", 1));
            }
        }
        return 1;
    }

    std::string runtime_configuration::get_cmd_line() const
    {
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx");
            if (NULL != sec) {
                return sec->get_entry("cmd_line", "");
            }
        }
        return "";
    }

    // Return the configured sizes of any of the know thread pools
    std::size_t runtime_configuration::get_thread_pool_size(char const* poolname) const
    {
        if (has_section("hpx.threadpools")) {
            util::section const* sec = get_section("hpx.threadpools");
            if (NULL != sec) {
                return boost::lexical_cast<std::size_t>(
                    sec->get_entry(std::string(poolname) + "_size", "2"));
            }
        }
        return 2;     // the default size for all pools is 2
    }

    // Return the endianess to be used for out-serialization
    std::string runtime_configuration::get_endian_out() const
    {
        if (has_section("hpx.parcel")) {
            util::section const* sec = get_section("hpx.parcel");
            if (NULL != sec) {
#ifdef BOOST_BIG_ENDIAN
                return sec->get_entry("endian_out", "big");
#else
                return sec->get_entry("endian_out", "little");
#endif
            }
        }
#ifdef BOOST_BIG_ENDIAN
        return "big"
#else
        return "little";
#endif
    }

    // Will return the stack size to use for all HPX-threads.
    std::ptrdiff_t runtime_configuration::init_stack_size(
        char const* entryname, char const* defaultvaluestr,
        std::ptrdiff_t defaultvalue) const
    {
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx.stacks");
            if (NULL != sec) {
                std::string entry = sec->get_entry(entryname, defaultvaluestr);
                std::ptrdiff_t val = defaultvalue;

                namespace qi = boost::spirit::qi;
                qi::parse(entry.begin(), entry.end(),
                    "0x" >> qi::hex | "0" >> qi::oct | qi::int_, val);
                return val;
            }
        }
        return defaultvalue;
    }

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)
    bool runtime_configuration::init_use_stack_guard_pages() const
    {
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx.stacks");
            if (NULL != sec) {
                std::string entry = sec->get_entry("use_guard_pages", "1");
                return (entry != "0") ? true : false;
            }
        }
        return true;    // default is true
    }
#endif

    std::ptrdiff_t runtime_configuration::init_small_stack_size() const
    {
        return init_stack_size("small_size",
            BOOST_PP_STRINGIZE(HPX_SMALL_STACK_SIZE), HPX_SMALL_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_medium_stack_size() const
    {
        return init_stack_size("medium_size",
            BOOST_PP_STRINGIZE(HPX_MEDIUM_STACK_SIZE), HPX_MEDIUM_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_large_stack_size() const
    {
        return init_stack_size("large_size",
            BOOST_PP_STRINGIZE(HPX_LARGE_STACK_SIZE), HPX_LARGE_STACK_SIZE);
    }

    std::ptrdiff_t runtime_configuration::init_huge_stack_size() const
    {
        return init_stack_size("huge_size",
            BOOST_PP_STRINGIZE(HPX_HUGE_STACK_SIZE), HPX_HUGE_STACK_SIZE);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Return maximally allowed message size
    boost::uint64_t runtime_configuration::get_max_message_size() const
    {
        if (has_section("hpx")) {
            util::section const* sec = get_section("hpx.parcel");
            if (NULL != sec) {
                return boost::lexical_cast<boost::uint64_t>(
                    sec->get_entry("max_message_size", HPX_PARCEL_MAX_MESSAGE_SIZE));
            }
        }
        return HPX_PARCEL_MAX_MESSAGE_SIZE;    // default is 1GByte
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

    ///////////////////////////////////////////////////////////////////////////
    std::ptrdiff_t runtime_configuration::get_stack_size(
        threads::thread_stacksize stacksize) const
    {
        switch (stacksize) {
        case threads::thread_stacksize_medium:
            return medium_stacksize;

        case threads::thread_stacksize_large:
            return large_stacksize;

        case threads::thread_stacksize_huge:
            return huge_stacksize;

        case threads::thread_stacksize_nostack:
            return 0;       // stackless thread

        default:
        case threads::thread_stacksize_small:
            break;
        }
        return small_stacksize;
    }
}}

