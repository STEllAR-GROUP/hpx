//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/server/memory_block.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>    // EXPORTS get_next_id
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/coroutines/coroutine.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/state.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/version.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN64) && defined(_DEBUG) && !defined(HPX_HAVE_FIBER_BASED_COROUTINES)
#include <io.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure the system gets properly shut down while handling Ctrl-C and other
// system signals
#if defined(HPX_WINDOWS)

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_ATTRIBUTE_NORETURN void handle_termination(char const* reason)
    {
        if (get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        std::cerr
#if defined(HPX_HAVE_STACKTRACES)
            << "{stack-trace}: " << hpx::util::trace() << "\n"
#endif
            << "{what}: " << (reason ? reason : "Unknown reason") << "\n"
            << full_build_string();           // add full build information

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

#include <signal.h>
#include <stdlib.h>
#include <string.h>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_ATTRIBUTE_NORETURN void termination_handler(int signum)
    {
        if (signum != SIGINT &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        char* reason = strsignal(signum);
        std::cerr
#if defined(HPX_HAVE_STACKTRACES)
            << "{stack-trace}: " << hpx::util::trace() << "\n"
#endif
            << "{what}: " << (reason ? reason : "Unknown signal") << "\n"
            << full_build_string();           // add full build information

        std::abort();
    }
}

#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // There is no need to protect these global from thread concurrent access
    // as they are access during early startup only.
    std::vector<hpx::util::tuple<char const*, char const*> >
        message_handler_registrations;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void HPX_CDECL new_handler()
    {
        HPX_THROW_EXCEPTION(out_of_memory, "new_handler",
            "new allocator failed to allocate memory");
    }

    void set_error_handlers()
    {
#if defined(HPX_WINDOWS)
        // Set console control handler to allow server to be stopped.
        SetConsoleCtrlHandler(hpx::termination_handler, TRUE);
#else
        struct sigaction new_action;
        new_action.sa_handler = hpx::termination_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags = 0;

        sigaction(SIGINT, &new_action, nullptr);  // Interrupted
        sigaction(SIGBUS, &new_action, nullptr);  // Bus error
        sigaction(SIGFPE, &new_action, nullptr);  // Floating point exception
        sigaction(SIGILL, &new_action, nullptr);  // Illegal instruction
        sigaction(SIGPIPE, &new_action, nullptr); // Bad pipe
        sigaction(SIGSEGV, &new_action, nullptr); // Segmentation fault
        sigaction(SIGSYS, &new_action, nullptr);  // Bad syscall
#endif

        std::set_new_handler(hpx::new_handler);
    }


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

    namespace strings
    {
        char const* const runtime_state_names[] =
        {
            "state_invalid",      // -1
            "state_initialized",  // 0
            "state_pre_startup",  // 1
            "state_startup",      // 2
            "state_pre_main",     // 3
            "state_starting",     // 4
            "state_running",      // 5
            "state_suspended",    // 6
            "state_pre_shutdown", // 7
            "state_shutdown",     // 8
            "state_stopping",     // 9
            "state_terminating",  // 10
            "state_stopped"       // 11
        };
    }

    char const* get_runtime_state_name(state st)
    {
        if (st < state_invalid || st >= last_valid_runtime_state)
            return "invalid (value out of bounds)";
        return strings::runtime_state_names[st+1];
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(util::runtime_configuration & rtcfg
          , threads::policies::init_affinity_data const& affinity_init)
      : ini_(rtcfg),
        instance_number_(++instance_number_counter_),
        thread_support_(new util::thread_mapper),
        affinity_init_(affinity_init),
        topology_(threads::create_topology()),
        state_(state_invalid),
        memory_(new components::server::memory),
        runtime_support_(new components::server::runtime_support(ini_))
    {
        LPROGRESS_;

        // initialize our TSS
        runtime::init_tss();
        util::reinit_construct();       // call only after TLS was initialized

        counters_.reset(new performance_counters::registry());
    }

    runtime::~runtime()
    {
        // allow to reuse instance number if this was the only instance
        if (0 == instance_number_counter_)
            --instance_number_counter_;
    }

    void runtime::set_state(state s)
    {
        LPROGRESS_ << get_runtime_state_name(s);
        state_.store(s);
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::atomic<int> runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////
    util::thread_specific_ptr<runtime*, runtime::tls_tag> runtime::runtime_;
    util::thread_specific_ptr<std::string, runtime::tls_tag> runtime::thread_name_;
    util::thread_specific_ptr<std::uint64_t, runtime::tls_tag> runtime::uptime_;

    void runtime::init_tss()
    {
        // initialize our TSS
        if (nullptr == runtime::runtime_.get())
        {
            HPX_ASSERT(nullptr == threads::thread_self::get_self());

            runtime::runtime_.reset(new runtime* (this));
            runtime::uptime_.reset(new std::uint64_t);
            *runtime::uptime_.get() = util::high_resolution_clock::now();

            threads::thread_self::init_self();
        }
    }

    void runtime::deinit_tss()
    {
        // reset our TSS
        threads::thread_self::reset_self();
        runtime::uptime_.reset();
        runtime::runtime_.reset();
        util::reset_held_lock_data();

        threads::reset_continuation_recursion_count();
    }

    std::string runtime::get_thread_name()
    {
        std::string const* str = runtime::thread_name_.get();
        return str ? *str : "<unknown>";
    }

    std::uint64_t runtime::get_system_uptime()
    {
        std::int64_t diff =
            util::high_resolution_clock::now() - *runtime::uptime_.get();
        return diff < 0LL ? 0ULL : static_cast<std::uint64_t>(diff);
    }

    performance_counters::registry& runtime::get_counter_registry()
    {
        return *counters_;
    }

    performance_counters::registry const& runtime::get_counter_registry() const
    {
        return *counters_;
    }

    util::thread_mapper& runtime::get_thread_mapper()
    {
        return *thread_support_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void runtime::register_query_counters(
        std::shared_ptr<util::query_counters> const& active_counters)
    {
        active_counters_ = active_counters;
    }

    void runtime::start_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->start_counters(ec);
    }

    void runtime::stop_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->stop_counters(ec);
    }

    void runtime::reset_active_counters(error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->reset_counters(ec);
    }

    void runtime::evaluate_active_counters(bool reset,
        char const* description, error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->evaluate_counters(reset, description, ec);
    }

    void runtime::stop_evaluating_counters()
    {
        if (active_counters_.get())
            active_counters_->stop_evaluating_counters();
    }

    void runtime::register_message_handler(char const* message_handler_type,
        char const* action, error_code& ec)
    {
        return runtime_support_->register_message_handler(
            message_handler_type, action, ec);
    }

    parcelset::policies::message_handler* runtime::create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        return runtime_support_->create_message_handler(message_handler_type,
            action, pp, num_messages, interval, ec);
    }

    serialization::binary_filter* runtime::create_binary_filter(
        char const* binary_filter_type, bool compress,
        serialization::binary_filter* next_filter, error_code& ec)
    {
        return runtime_support_->create_binary_filter(binary_filter_type,
            compress, next_filter, ec);
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
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // stddev counter
            { "/statistics/stddev", performance_counters::counter_aggregating,
              "returns the standard deviation value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/stddev",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // rolling_averaging counter
            { "/statistics/rolling_average", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/rolling_averaging",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // median counter
            { "/statistics/median", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/median",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // max counter
            { "/statistics/max", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/max",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // min counter
            { "/statistics/min", performance_counters::counter_aggregating,
              "returns the averaged value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/min",
              HPX_PERFORMANCE_COUNTER_V1,
               &performance_counters::detail::statistics_counter_creator,
               &performance_counters::default_counter_discoverer,
              ""
            },

            // uptime counters
            { "/runtime/uptime", performance_counters::counter_elapsed_time,
              "returns the up time of the runtime instance for the referenced "
              "locality",
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
            },

            // action invocation counters
            { "/runtime/count/action-invocation", performance_counters::counter_raw,
              "returns the number of (local) invocations of a specific action "
              "on this locality (the action type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::local_action_invocation_counter_creator,
              &performance_counters::local_action_invocation_counter_discoverer,
              ""
            },

            { "/runtime/count/remote-action-invocation",
              performance_counters::counter_raw,
              "returns the number of (remote) invocations of a specific action "
              "on this locality (the action type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::remote_action_invocation_counter_creator,
              &performance_counters::remote_action_invocation_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            statistic_counter_types,
            sizeof(statistic_counter_types)/sizeof(statistic_counter_types[0]));

        performance_counters::generic_counter_type_data arithmetic_counter_types[] =
        {
            // adding counter
            { "/arithmetics/add", performance_counters::counter_aggregating,
              "returns the sum of the values of the specified base counters; "
              "pass required base counters as the parameters: "
              "/arithmetics/add@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // minus counter
            { "/arithmetics/subtract", performance_counters::counter_aggregating,
              "returns the difference of the values of the specified base counters; "
              "pass the required base counters as the parameters: "
              "/arithmetics/subtract@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // multiply counter
            { "/arithmetics/multiply", performance_counters::counter_aggregating,
              "returns the product of the values of the specified base counters; "
              "pass the required base counters as the parameters: "
              "/arithmetics/multiply@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // divide counter
            { "/arithmetics/divide", performance_counters::counter_aggregating,
              "returns the result of division of the values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/divide@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
        };
        performance_counters::install_counter_types(
            arithmetic_counter_types,
            sizeof(arithmetic_counter_types)/sizeof(arithmetic_counter_types[0]));
    }

    std::uint32_t runtime::assign_cores(std::string const& locality_basename,
        std::uint32_t cores_needed)
    {
        std::lock_guard<compat::mutex> l(mtx_);

        used_cores_map_type::iterator it = used_cores_map_.find(locality_basename);
        if (it == used_cores_map_.end())
        {
            used_cores_map_.insert(
                used_cores_map_type::value_type(locality_basename, cores_needed));
            return 0;
        }

        std::uint32_t current = (*it).second;
        (*it).second += cores_needed;
        return current;
    }

    std::uint32_t runtime::assign_cores()
    {
        // initialize thread affinity settings in the scheduler
        if (affinity_init_.used_cores_ == 0) {
            // correct used_cores from config data if appropriate
            affinity_init_.used_cores_ = std::size_t(
                this->get_config().get_first_used_core());
        }

        return static_cast<std::uint32_t>(
            this->get_thread_manager().init(affinity_init_));
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime& get_runtime()
    {
        HPX_ASSERT(nullptr != runtime::runtime_.get());   // should have been initialized
        return **runtime::runtime_;
    }

    runtime* get_runtime_ptr()
    {
        runtime** rt = runtime::runtime_.get();
        return rt ? *rt : nullptr;
    }

    naming::gid_type const & get_locality()
    {
        return get_runtime().get_agas_client().get_local_locality();
    }

    /// Register the current kernel thread with HPX, this should be done once
    /// for each external OS-thread intended to invoke HPX functionality.
    /// Calling this function more than once will silently fail
    /// (will return false).
    bool register_thread(runtime* rt, char const* name, error_code& ec)
    {
        HPX_ASSERT(rt);
        return rt->register_thread(name, 0, true, ec);
    }

    /// Unregister the thread from HPX, this should be done once in
    /// the end before the external thread exists.
    void unregister_thread(runtime* rt)
    {
        HPX_ASSERT(rt);
        rt->unregister_thread();
    }

    void report_error(std::size_t num_thread, boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(state_running))
        {
            hpx::runtime* rt = hpx::get_runtime_ptr();
            if (rt)
                rt->report_error(num_thread, e);
            else
                detail::report_exception_and_terminate(e);
            return;
        }

        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    void report_error(boost::exception_ptr const& e)
    {
        // Early and late exceptions
        if (!threads::threadmanager_is(state_running))
        {
            hpx::runtime* rt = hpx::get_runtime_ptr();
            if (rt)
                rt->report_error(std::size_t(-1), e);
            else
                detail::report_exception_and_terminate(e);
            return;
        }

        std::size_t num_thread = hpx::get_worker_thread_num();
        hpx::applier::get_applier().get_thread_manager().report_error(num_thread, e);
    }

    bool register_on_exit(util::function_nonser<void()> const& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return false;

        rt->on_exit(f);
        return true;
    }

    std::size_t get_runtime_instance_number()
    {
        runtime* rt = get_runtime_ptr();
        return (nullptr == rt) ? 0 : rt->get_instance_number();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string get_config_entry(std::string const& key, std::string const& dflt)
    {
        if (nullptr == get_runtime_ptr())
            return dflt;
        return get_runtime().get_config().get_entry(key, dflt);
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return std::to_string(dflt);
        return get_runtime().get_config().get_entry(key, dflt);
    }

    // set entries
    void set_config_entry(std::string const& key, std::string const& value)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return;
        return rt->get_config().add_entry(key, value);
    }

    void set_config_entry(std::string const& key, std::size_t value)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return;
        return rt->get_config().add_entry(key, std::to_string(value));
    }

    void set_config_entry_callback(std::string const& key,
        util::function_nonser<
            void(std::string const&, std::string const&)
        > const& callback)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
            return;
        return rt->get_config().add_notification_callback(key, callback);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Helpers
    naming::id_type find_here(error_code& ec)
    {
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_here",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        return naming::id_type(hpx::applier::get_applier().get_raw_locality(ec),
            naming::id_type::unmanaged);
    }

    naming::id_type find_root_locality(error_code& ec)
    {
        runtime* rt = hpx::get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_root_locality",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        naming::gid_type console_locality;
        if (!rt->get_agas_client().get_console_locality(console_locality))
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_root_locality",
                "the root locality is not available at this time");
            return naming::invalid_id;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return naming::id_type(console_locality, naming::id_type::unmanaged);
    }

    std::vector<naming::id_type>
    find_all_localities(components::component_type type, error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_all_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_localities(locality_ids, type, ec);
        return locality_ids;
    }

    std::vector<naming::id_type> find_all_localities(error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_all_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_localities(locality_ids, ec);
        return locality_ids;
    }

    std::vector<naming::id_type>
    find_remote_localities(components::component_type type, error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_remote_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_remote_localities(locality_ids, type, ec);
        return locality_ids;
    }

    std::vector<naming::id_type> find_remote_localities(error_code& ec)
    {
        std::vector<naming::id_type> locality_ids;
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_remote_localities",
                "the runtime system is not available at this time");
            return locality_ids;
        }

        hpx::applier::get_applier().get_remote_localities(locality_ids,
            components::component_invalid, ec);

        return locality_ids;
    }

    // find a locality supporting the given component
    naming::id_type find_locality(components::component_type type, error_code& ec)
    {
        if (nullptr == hpx::applier::get_applier_ptr())
        {
            HPX_THROWS_IF(ec, invalid_status, "hpx::find_locality",
                "the runtime system is not available at this time");
            return naming::invalid_id;
        }

        std::vector<naming::id_type> locality_ids;
        hpx::applier::get_applier().get_localities(locality_ids, type, ec);

        if (ec || locality_ids.empty())
            return naming::invalid_id;

        // chose first locality to host the object
        return locality_ids.front();
    }

    /// \brief Return the number of localities which are currently registered
    ///        for the running application.
    std::uint32_t get_num_localities(hpx::launch::sync_policy, error_code& ec)
    {
        if (nullptr == hpx::get_runtime_ptr())
            return 0;

        return get_runtime().get_agas_client().get_num_localities(ec);
    }

    std::uint32_t get_initial_num_localities()
    {
        if (nullptr == hpx::get_runtime_ptr())
            return 0;

        return get_runtime().get_config().get_num_localities();
    }

    std::uint32_t get_num_localities(hpx::launch::sync_policy,
        components::component_type type, error_code& ec)
    {
        if (nullptr == hpx::get_runtime_ptr())
            return 0;

        return get_runtime().get_agas_client().get_num_localities(type, ec);
    }

    lcos::future<std::uint32_t> get_num_localities()
    {
        if (nullptr == hpx::get_runtime_ptr())
            return lcos::make_ready_future<std::uint32_t>(0);

        return get_runtime().get_agas_client().get_num_localities_async();
    }

    lcos::future<std::uint32_t> get_num_localities(
        components::component_type type)
    {
        if (nullptr == hpx::get_runtime_ptr())
            return lcos::make_ready_future<std::uint32_t>(0);

        return get_runtime().get_agas_client().get_num_localities_async(type);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        naming::gid_type get_next_id(std::size_t count)
        {
            if (nullptr == get_runtime_ptr())
                return naming::invalid_gid;

            return get_runtime().get_next_id(count);
        }

        ///////////////////////////////////////////////////////////////////////////
        void dijkstra_make_black()
        {
            get_runtime_support_ptr()->dijkstra_make_black();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_os_thread_count()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(
                invalid_status,
                "hpx::get_os_thread_count()",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }
        return rt->get_config().get_os_thread_count();
    }

    std::size_t get_os_thread_count(threads::executor const& exec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(
                invalid_status,
                "hpx::get_os_thread_count(exec)",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }

        if (!exec)
            return rt->get_config().get_os_thread_count();

        error_code ec(lightweight);
        return exec.executor_data_->get_policy_element(
            threads::detail::current_concurrency, ec);
    }

    std::size_t get_worker_thread_num()
    {
        return get_worker_thread_num(throws);
    }

    std::size_t get_worker_thread_num(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROWS_IF(
                ec, invalid_status,
                "hpx::get_worker_thread_num",
                "the runtime system has not been initialized yet");
            return std::size_t(-1);
        }
        return rt->get_thread_manager().get_worker_thread_num();
    }

    std::size_t get_num_worker_threads()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(
                invalid_status,
                "hpx::get_num_worker_threads",
                "the runtime system has not been initialized yet");
            return std::size_t(0);
        }

        error_code ec(lightweight);
        return static_cast<std::size_t>(
            rt->get_agas_client().get_num_overall_threads(ec));
    }

    bool is_scheduler_numa_sensitive()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr == rt)
        {
            HPX_THROW_EXCEPTION(
                invalid_status,
                "hpx::is_scheduler_numa_sensitive",
                "the runtime system has not been initialized yet");
            return false;
        }

        bool numa_sensitive = false;
        if (std::size_t(-1) !=
            rt->get_thread_manager().get_worker_thread_num(&numa_sensitive))
            return numa_sensitive;
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool keep_factory_alive(components::component_type type)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            return rt->keep_factory_alive(type);

        HPX_THROW_EXCEPTION(
            invalid_status,
            "hpx::keep_factory_alive",
            "the runtime system has not been initialized yet");
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
        if (nullptr != rt)
            return rt->get_state() == state_running;
        return false;
    }

    bool is_stopped()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            return rt->get_state() == state_stopped;
        return true;        // assume stopped
    }

    bool is_stopped_or_shutting_down()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            state st = rt->get_state();
            return st >= state_shutdown;
        }
        return true;        // assume stopped
    }

    bool HPX_EXPORT is_starting()
    {
        runtime* rt = get_runtime_ptr();
        return nullptr != rt ? rt->get_state() <= state_startup : true;
    }

    bool HPX_EXPORT is_pre_startup()
    {
        runtime* rt = get_runtime_ptr();
        return nullptr != rt ? rt->get_state() < state_startup : true;
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
namespace hpx { namespace parcelset
{
    bool do_background_work(std::size_t num_thread)
    {
        return get_runtime().get_parcel_handler().do_background_work(num_thread);
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
        if (stacksize == threads::thread_stacksize_current)
            return threads::get_self_stacksize();

        return get_runtime().get_config().get_stack_size(stacksize);
    }

    HPX_API_EXPORT void reset_thread_distribution()
    {
        get_runtime().get_thread_manager().reset_thread_distribution();
    }

    HPX_API_EXPORT void set_scheduler_mode(threads::policies::scheduler_mode m)
    {
        get_runtime().get_thread_manager().set_scheduler_mode(m);
    }

    HPX_API_EXPORT threads::mask_cref_type get_pu_mask(
        threads::topology& topo, std::size_t thread_num)
    {
        return get_runtime().get_thread_manager().get_pu_mask(topo, thread_num);
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    std::uint32_t get_locality_id(error_code& ec)
    {
        return agas::get_locality_id(ec);
    }

    std::string get_thread_name()
    {
        return runtime::get_thread_name();
    }

    struct itt_domain_tag {};
    static util::thread_specific_ptr<util::itt::domain, itt_domain_tag> d;

    util::itt::domain const& get_thread_itt_domain()
    {
        if (nullptr == d.get())
            d.reset(new util::itt::domain(get_thread_name().c_str()));

        return *d;
    }

    std::uint64_t get_system_uptime()
    {
        return runtime::get_system_uptime();
    }

    util::runtime_configuration const& get_config()
    {
        return get_runtime().get_config();
    }

    hpx::util::io_service_pool* get_thread_pool(
        char const* name, char const* name_suffix)
    {
        std::string full_name(name);
        full_name += name_suffix;
        return get_runtime().get_thread_pool(full_name.c_str());
    }

    ///////////////////////////////////////////////////////////////////////////
    void start_active_counters(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            rt->start_active_counters(ec);
        }
        else {
            HPX_THROWS_IF(ec, invalid_status, "start_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void stop_active_counters(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            rt->stop_active_counters(ec);
        }
        else {
            HPX_THROWS_IF(ec, invalid_status, "stop_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void reset_active_counters(error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            rt->reset_active_counters(ec);
        }
        else {
            HPX_THROWS_IF(ec, invalid_status, "reset_active_counters",
                "the runtime system is not available at this time");
        }
    }

    void evaluate_active_counters(bool reset, char const* description, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            rt->evaluate_active_counters(reset, description, ec);
        }
        else {
            HPX_THROWS_IF(ec, invalid_status, "evaluate_active_counters",
                "the runtime system is not available at this time");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Create an instance of a message handler plugin
    void register_message_handler(char const* message_handler_type,
        char const* action, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            return rt->register_message_handler(message_handler_type, action, ec);
        }

        // store the request for later
        message_handler_registrations.push_back(
            hpx::util::make_tuple(message_handler_type, action));
    }

    parcelset::policies::message_handler* create_message_handler(
        char const* message_handler_type, char const* action,
        parcelset::parcelport* pp, std::size_t num_messages,
        std::size_t interval, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            return rt->create_message_handler(message_handler_type, action,
                pp, num_messages, interval, ec);
        }

        HPX_THROWS_IF(ec, invalid_status, "create_message_handler",
            "the runtime system is not available at this time");
        return nullptr;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Create an instance of a binary filter plugin
    serialization::binary_filter* create_binary_filter(char const* binary_filter_type,
        bool compress, serialization::binary_filter* next_filter, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            return rt->create_binary_filter
                    (binary_filter_type, compress, next_filter, ec);

        HPX_THROWS_IF(ec, invalid_status, "create_binary_filter",
            "the runtime system is not available at this time");
        return nullptr;
    }

    // helper function to stop evaluating counters during shutdown
    void stop_evaluating_counters()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) rt->stop_evaluating_counters();
    }
}

