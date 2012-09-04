//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <iostream>
#include <vector>

#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>

#if defined(HPX_HAVE_HWLOC)
    #include <hpx/runtime/threads/policies/hwloc_topology.hpp>
#elif defined(BOOST_WINDOWS)
    #include <hpx/runtime/threads/policies/windows_topology.hpp>
#elif defined(__APPLE__)
    #include <hpx/runtime/threads/policies/macosx_topology.hpp>
#elif defined(__linux__)
    #include <hpx/runtime/threads/policies/linux_topology.hpp>
#else
    #include <hpx/runtime/threads/policies/noop_topology.hpp>
#endif

#include <hpx/util/coroutine/detail/coroutine_impl_impl.hpp>
#if defined(HPX_HAVE_STACKTRACES)
#include <boost/backtrace.hpp>
#endif

#if defined(_WIN64) && defined(_DEBUG) && !defined(HPX_COROUTINE_USE_FIBERS)
#include <io.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure the system gets properly shut down while handling Ctrl-C and other
// system signals
#if defined(BOOST_WINDOWS)

namespace hpx
{
    void handle_termination(char const* reason)
    {
        std::cerr << "Received " << (reason ? reason : "unknown signal")
#if defined(HPX_HAVE_STACKTRACES)
                  << ", " << hpx::detail::backtrace()
#else
                  << "."
#endif
                  << std::endl;
        std::abort();
    }

    HPX_EXPORT BOOL WINAPI termination_handler(DWORD ctrl_type)
    {
        switch (ctrl_type) {
        case CTRL_C_EVENT:
            handle_termination("Ctrl-C");
            return TRUE;

        case CTRL_BREAK_EVENT:
            handle_termination("Ctrl-Break");
            return TRUE;

        case CTRL_CLOSE_EVENT:
            handle_termination("Ctrl-Close");
            return TRUE;

        case CTRL_LOGOFF_EVENT:
            handle_termination("Logoff");
            return TRUE;

        case CTRL_SHUTDOWN_EVENT:
            handle_termination("Shutdown");
            return TRUE;

        default:
            break;
        }
        return FALSE;
    }
}

#else

//#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

namespace hpx
{
    HPX_EXPORT void termination_handler(int signum)
    {
        char* c = strsignal(signum);
        std::cerr << "Received " << (c ? c : "unknown signal")
#if defined(HPX_HAVE_STACKTRACES)
                  << ", " << hpx::detail::backtrace()
#else
                  << "."
#endif
                  << std::endl;
        std::abort();
    }
}

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace strings
    {
        char const* const runtime_mode_names[] =
        {
            "invalid",    // -1
            "console",    // 0
            "worker",     // 1
            "connect",    // 2
            "default",    // 3
        };
    }

    char const* get_runtime_mode_name(runtime_mode state)
    {
        if (state < runtime_mode_invalid || state >= runtime_mode_last)
            return "invalid (value out of bounds)";
        return strings::runtime_mode_names[state+1];
    }

    runtime_mode get_runtime_mode_from_name(std::string const& mode)
    {
        for (std::size_t i = 0; i < runtime_mode_last; ++i) {
            if (mode == strings::runtime_mode_names[i])
                return static_cast<runtime_mode>(i-1);
        }
        return runtime_mode_invalid;
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(naming::resolver_client& agas_client,
            util::runtime_configuration& rtcfg)
      : ini_(rtcfg),
        instance_number_(++instance_number_counter_),
        topology_(
#if defined(HPX_HAVE_HWLOC)
            new threads::hwloc_topology
#elif defined(BOOST_WINDOWS)
            new threads::windows_topology
#elif defined(__APPLE__)
            new threads::macosx_topology
#elif defined(__linux__)
            new threads::linux_topology
#else
            new threads::noop_topology
#endif
        ),
        state_(state_invalid)
    {
        // initialize our TSS
        runtime::init_tss();

        counters_.reset(new performance_counters::registry(agas_client));
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::atomic<int> runtime::instance_number_counter_(-1);


    ///////////////////////////////////////////////////////////////////////////
    hpx::util::thread_specific_ptr<runtime *, runtime::tls_tag> runtime::runtime_;
    hpx::util::thread_specific_ptr<std::string, runtime::tls_tag> runtime::thread_name_;

    void runtime::init_tss()
    {
        // initialize our TSS
        if (NULL == runtime::runtime_.get())
        {
            BOOST_ASSERT(NULL == threads::coroutine_type::impl_type::get_self());

            runtime::runtime_.reset(new runtime* (this));
            threads::coroutine_type::impl_type::init_self();
        }
    }

    void runtime::deinit_tss()
    {
        // reset our TSS
        threads::coroutine_type::impl_type::reset_self();
        runtime::runtime_.reset();
    }

    /// \brief Register all performance counter types related to this runtime
    ///        instance
    void runtime::register_counter_types()
    {
        performance_counters::generic_counter_type_data statistic_counter_types[] =
        {
            // averaging counter
            { "/statistics/average", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/average",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::aggregating_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // max counter
            { "/statistics/max", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/max",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::aggregating_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // min counter
            { "/statistics/min", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/min",
              HPX_PERFORMANCE_COUNTER_V1,
               &performance_counters::detail::aggregating_counter_creator,
               &performance_counters::default_counter_discoverer,
              ""
            },

            // uptime counters
            { "/runtime/uptime", performance_counters::counter_elapsed_time,
              "returns the up time of the runtime instance for the referenced locality",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::uptime_counter_creator,
              &performance_counters::locality_counter_discoverer,
              "s"    // unit of measure is seconds
            },

            // component instance counters
            { "/runtime/count/component", performance_counters::counter_raw,
              "returns the number of component instances currently alive on "
              "this locality (the component type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::component_instance_counter_creator,
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            statistic_counter_types,
            sizeof(statistic_counter_types)/sizeof(statistic_counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime& get_runtime()
    {
        BOOST_ASSERT(NULL != runtime::runtime_.get());   // should have been initialized
        return **runtime::runtime_;
    }

    runtime* get_runtime_ptr()
    {
        runtime** rt = runtime::runtime_.get();
        return rt ? *rt : NULL;
    }

    naming::locality const& get_locality()
    {
        return get_runtime().here();
    }

    void report_error(std::size_t num_thread, boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(running))
        {
            detail::report_exception_and_terminate(e);
            return;
        }

        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    void report_error(boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(running))
        {
            detail::report_exception_and_terminate(e);
            return;
        }

        std::size_t num_thread = hpx::threads::threadmanager_base::get_worker_thread_num();
        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    bool register_on_exit(HPX_STD_FUNCTION<void()> f)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return false;

        rt->on_exit(f);
        return true;
    }

    std::size_t get_runtime_instance_number()
    {
        runtime* rt = get_runtime_ptr();
        return (NULL == rt) ? 0 : rt->get_instance_number();
    }

    std::string get_config_entry(std::string const& key, std::string const& dflt)
    {
        if (NULL == get_runtime_ptr())
            return "";
        return get_runtime().get_config().get_entry(key, dflt);
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        if (NULL == get_runtime_ptr())
            return "";
        return get_runtime().get_config().get_entry(key, dflt);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    naming::id_type find_here()
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return naming::invalid_id;

        return naming::id_type(hpx::applier::get_applier().get_raw_locality(),
            naming::id_type::unmanaged);
    }

    std::vector<naming::id_type>
    find_all_localities(components::component_type type)
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_localities(locality_ids, type);
        return locality_ids;
    }

    std::vector<naming::id_type> find_all_localities()
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_localities(locality_ids);
        return locality_ids;
    }

    std::vector<naming::id_type>
    find_remote_localities(components::component_type type)
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_remote_localities(locality_ids, type);
        return locality_ids;
    }

    std::vector<naming::id_type> find_remote_localities()
    {
        std::vector<naming::id_type> locality_ids;
        if (NULL != hpx::applier::get_applier_ptr())
            hpx::applier::get_applier().get_remote_localities(locality_ids);
        return locality_ids;
    }

    // find a locality supporting the given component
    naming::id_type find_locality(components::component_type type)
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return naming::invalid_id;

        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type);

        if (locality_ids.empty()) {
            HPX_THROW_EXCEPTION(hpx::bad_component_type, "find_locality",
                "no locality supporting sheneos configuration component found");
            return naming::invalid_id;
        }

        // chose first locality to host the object
        return locality_ids.front();
    }

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    boost::uint32_t get_num_localities()
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return 0;

        // FIXME: this is overkill
        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids);
        return static_cast<boost::uint32_t>(locality_ids.size());
    }

    boost::uint32_t get_num_localities(components::component_type type)
    {
        if (NULL == hpx::applier::get_applier_ptr())
            return 0;

        // FIXME: this is overkill
        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type);
        return static_cast<boost::uint32_t>(locality_ids.size());
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        naming::gid_type get_next_id()
        {
            if (NULL == hpx::applier::get_applier_ptr())
                return naming::invalid_gid;

            return get_runtime().get_next_id();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_os_thread_count()
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return 0;
        return rt->get_config().get_os_thread_count();
    }

    std::size_t get_worker_thread_num()
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return std::size_t(-1);
        return rt->get_thread_manager().get_worker_thread_num();
    }

    bool is_scheduler_numa_sensitive()
    {
        runtime* rt = get_runtime_ptr();
        if (NULL == rt)
            return false;

        bool numa_sensitive = false;
        if (std::size_t(-1) != rt->get_thread_manager().get_worker_thread_num(&numa_sensitive))
            return numa_sensitive;
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool keep_factory_alive(components::component_type type)
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt)
            return rt->keep_factory_alive(type);
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    components::server::runtime_support* get_runtime_support_ptr()
    {
        return reinterpret_cast<components::server::runtime_support*>(
            get_runtime().get_runtime_support_lva());
    }

    ///////////////////////////////////////////////////////////////////////////
    bool is_running()
    {
        runtime* rt = get_runtime_ptr();
        if (NULL != rt)
            return rt->get_state() == runtime::state_running;
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    std::string expand(std::string const& in)
    {
        return get_runtime().get_config().expand(in);
    }

    void expand(std::string& in)
    {
        get_runtime().get_config().expand(in, std::string::size_type(-1));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming
{
    // shortcut for get_runtime().get_agas_client()
    resolver_client& get_agas_client()
    {
        return get_runtime().get_agas_client();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    // shortcut for get_applier().get_thread_manager()
    threadmanager_base& get_thread_manager()
    {
        return get_runtime().get_thread_manager();
    }

    // shortcut for runtime_configuration::get_default_stack_size
    std::ptrdiff_t get_default_stack_size()
    {
        return get_runtime().get_config().get_default_stack_size();
    }

    // shortcut for runtime_configuration::get_stack_size
    std::ptrdiff_t get_stack_size(threads::thread_stacksize stacksize)
    {
        return get_runtime().get_config().get_stack_size(stacksize);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    boost::uint32_t get_locality_id(error_code& ec)
    {
        return agas::get_locality_id(ec);
    }
}

