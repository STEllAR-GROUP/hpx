//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/concurrency/thread_name.hpp>
#include <hpx/custom_exception_info.hpp>
#include <hpx/errors.hpp>
#include <hpx/static_reinit/static_reinit.hpp>
#include <hpx/logging.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>    // EXPORTS get_next_id
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/thread_hooks.hpp>
#include <hpx/coroutines/coroutine.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/topology/topology.hpp>
#include <hpx/state.hpp>
#include <hpx/timing/high_resolution_clock.hpp>
#include <hpx/debugging/backtrace.hpp>
#include <hpx/command_line_handling.hpp>
#include <hpx/util/debugging.hpp>
#include <hpx/util/from_string.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/thread_mapper.hpp>
#include <hpx/version.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
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
    void handle_termination(char const* reason)
    {
        if (get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        if (get_config_entry("hpx.diagnostics_on_terminate", "1") == "1")
        {
            std::cerr
#if defined(HPX_HAVE_STACKTRACES)
                << "{stack-trace}: " << hpx::util::trace() << "\n"
#endif
                << "{what}: " << (reason ? reason : "Unknown reason") << "\n"
                << full_build_string();           // add full build information
        }
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
    HPX_EXPORT HPX_NORETURN void termination_handler(int signum)
    {
        if (signum != SIGINT &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        if (get_config_entry("hpx.diagnostics_on_terminate", "1") == "1")
        {
            char* reason = strsignal(signum);
            std::cerr
#if defined(HPX_HAVE_STACKTRACES)
                << "{stack-trace}: " << hpx::util::trace() << "\n"
#endif
                << "{what}: " << (reason ? reason : "Unknown signal") << "\n"
                << full_build_string();           // add full build information
        }
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
#if defined(HPX_HAVE_NETWORKING)
    std::vector<hpx::util::tuple<char const*, char const*>>&
        get_message_handler_registrations()
    {
        static std::vector<hpx::util::tuple<char const*, char const*>>
            message_handler_registrations;
        return message_handler_registrations;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT void HPX_CDECL new_handler()
    {
        HPX_THROW_EXCEPTION(out_of_memory, "new_handler",
            "new allocator failed to allocate memory");
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Sometimes the HPX library gets simply unloaded as a result of some
        // extreme error handling. Avoid hangs in the end by setting a flag.
        static bool exit_called = false;

        void on_exit() noexcept
        {
            exit_called = true;
        }

        void on_abort(int) noexcept
        {
            exit_called = true;
            std::exit(-1);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
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
            "state_pre_sleep",    // 7
            "state_sleeping",     // 8
            "state_pre_shutdown", // 9
            "state_shutdown",     // 10
            "state_stopping",     // 11
            "state_terminating",  // 12
            "state_stopped"       // 13
        };
    }

    char const* get_runtime_state_name(state st)
    {
        if (st < state_invalid || st >= last_valid_runtime_state)
            return "invalid (value out of bounds)";
        return strings::runtime_state_names[st+1];
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::policies::callback_notifier::on_startstop_type global_on_start_func;
    threads::policies::callback_notifier::on_startstop_type global_on_stop_func;
    threads::policies::callback_notifier::on_error_type global_on_error_func;

    ///////////////////////////////////////////////////////////////////////////
    runtime::runtime(util::runtime_configuration & rtcfg)
      : ini_(rtcfg),
        instance_number_(++instance_number_counter_),
        thread_support_(new util::thread_mapper),
        topology_(resource::get_partitioner().get_topology()),
        state_(state_invalid),
        memory_(new components::server::memory),
        runtime_support_(new components::server::runtime_support(ini_)),
        on_start_func_(global_on_start_func),
        on_stop_func_(global_on_stop_func),
        on_error_func_(global_on_error_func)
    {
        LPROGRESS_;

        // initialize our TSS
        runtime::init_tss();
        util::reinit_construct();       // call only after TLS was initialized

        counters_ = std::make_shared<performance_counters::registry>();
    }

    runtime::~runtime()
    {
        // allow to reuse instance number if this was the only instance
        if (0 == instance_number_counter_)
            --instance_number_counter_;

        util::reinit_destruct();
        resource::detail::delete_partitioner();
    }

    void runtime::set_state(state s)
    {
        LPROGRESS_ << get_runtime_state_name(s);
        state_.store(s);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::atomic<int> runtime::instance_number_counter_(-1);

    ///////////////////////////////////////////////////////////////////////////

    namespace {
        std::uint64_t& runtime_uptime()
        {
            static thread_local std::uint64_t uptime;
            return uptime;
        }
    }

    void runtime::init_tss()
    {
        // initialize our TSS
        runtime*& runtime_ = get_runtime_ptr();
        if (nullptr == runtime_)
        {
            HPX_ASSERT(nullptr == threads::thread_self::get_self());

            runtime_ = this;
            runtime_uptime() = util::high_resolution_clock::now();
        }
    }

    void runtime::deinit_tss()
    {
        // reset our TSS
        runtime_uptime() = 0;
        get_runtime_ptr() = nullptr;
        threads::reset_continuation_recursion_count();
    }

    std::uint64_t runtime::get_system_uptime()
    {
        std::int64_t diff =
            util::high_resolution_clock::now() - runtime_uptime();
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

    void runtime::reinit_active_counters(bool reset, error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->reinit_counters(reset, ec);
    }

    void runtime::evaluate_active_counters(bool reset,
        char const* description, error_code& ec)
    {
        if (active_counters_.get())
            active_counters_->evaluate_counters(reset, description, true, ec);
    }

    void runtime::stop_evaluating_counters(bool terminate)
    {
        if (active_counters_.get())
            active_counters_->stop_evaluating_counters(terminate);
    }

#if defined(HPX_HAVE_NETWORKING)
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
#endif

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
              "returns the rolling average value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/rolling_averaging",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // rolling stddev counter
            { "/statistics/rolling_stddev", performance_counters::counter_aggregating,
              "returns the rolling standard deviation value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/rolling_stddev",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // median counter
            { "/statistics/median", performance_counters::counter_aggregating,
              "returns the median value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/median",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // max counter
            { "/statistics/max", performance_counters::counter_aggregating,
              "returns the maximum value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/max",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // min counter
            { "/statistics/min", performance_counters::counter_aggregating,
              "returns the minimum value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/min",
              HPX_PERFORMANCE_COUNTER_V1,
               &performance_counters::detail::statistics_counter_creator,
               &performance_counters::default_counter_discoverer,
              ""
            },

            // rolling max counter
            { "/statistics/rolling_max", performance_counters::counter_aggregating,
              "returns the rolling maximum value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/rolling_max",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::statistics_counter_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },

            // rolling min counter
            { "/statistics/rolling_min", performance_counters::counter_aggregating,
              "returns the rolling minimum value of its base counter over "
              "an arbitrary time line; pass required base counter as the instance "
              "name: /statistics{<base_counter_name>}/rolling_min",
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
            { "/runtime/count/action-invocation",
              performance_counters::counter_monotonically_increasing,
              "returns the number of (local) invocations of a specific action "
              "on this locality (the action type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::local_action_invocation_counter_creator,
              &performance_counters::local_action_invocation_counter_discoverer,
              ""
            },
#if defined(HPX_HAVE_NETWORKING)
            { "/runtime/count/remote-action-invocation",
              performance_counters::counter_monotonically_increasing,
              "returns the number of (remote) invocations of a specific action "
              "on this locality (the action type has to be specified as the "
              "counter parameter)",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::remote_action_invocation_counter_creator,
              &performance_counters::remote_action_invocation_counter_discoverer,
              ""
            }
#endif
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

            // arithmetics mean counter
            { "/arithmetics/mean", performance_counters::counter_aggregating,
              "returns the average value of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/mean@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // arithmetics variance counter
            { "/arithmetics/variance", performance_counters::counter_aggregating,
              "returns the standard deviation of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/variance@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // arithmetics median counter
            { "/arithmetics/median", performance_counters::counter_aggregating,
              "returns the median of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/median@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // arithmetics min counter
            { "/arithmetics/min", performance_counters::counter_aggregating,
              "returns the minimum value of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/min@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // arithmetics max counter
            { "/arithmetics/max", performance_counters::counter_aggregating,
              "returns the maximum value of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/max@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
              &performance_counters::default_counter_discoverer,
              ""
            },
            // arithmetics count counter
            { "/arithmetics/count", performance_counters::counter_aggregating,
              "returns the count value of all values of the specified "
              "base counters; pass the required base counters as the parameters: "
              "/arithmetics/count@<base_counter_name1>,<base_counter_name2>",
              HPX_PERFORMANCE_COUNTER_V1,
              &performance_counters::detail::arithmetics_counter_extended_creator,
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
        std::lock_guard<std::mutex> l(mtx_);

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
        // adjust thread assignments to allow for more than one locality per
        // node
        std::size_t first_core =
            static_cast<std::size_t>(this->get_config().get_first_used_core());
        std::size_t cores_needed =
            hpx::resource::get_partitioner().assign_cores(first_core);

        return static_cast<std::uint32_t>(cores_needed);
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::policies::callback_notifier::on_startstop_type
        runtime::on_start_func() const
    {
        return on_start_func_;
    }

    threads::policies::callback_notifier::on_startstop_type
        runtime::on_stop_func() const
    {
        return on_stop_func_;
    }

    threads::policies::callback_notifier::on_error_type
        runtime::on_error_func() const
    {
        return on_error_func_;
    }

    threads::policies::callback_notifier::on_startstop_type
    runtime::on_start_func(
        threads::policies::callback_notifier::on_startstop_type&& f)
    {
        threads::policies::callback_notifier::on_startstop_type newf =
            std::move(f);
        std::swap(on_start_func_, newf);
        return newf;
    }

    threads::policies::callback_notifier::on_startstop_type
    runtime::on_stop_func(
        threads::policies::callback_notifier::on_startstop_type&& f)
    {
        threads::policies::callback_notifier::on_startstop_type newf =
            std::move(f);
        std::swap(on_stop_func_, newf);
        return newf;
    }

    threads::policies::callback_notifier::on_error_type
    runtime::on_error_func(
        threads::policies::callback_notifier::on_error_type&& f)
    {
        threads::policies::callback_notifier::on_error_type newf =
            std::move(f);
        std::swap(on_error_func_, newf);
        return newf;
    }

    ///////////////////////////////////////////////////////////////////////////
    threads::policies::callback_notifier::on_startstop_type
        get_thread_on_start_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_start_func();
        }
        else
        {
            return global_on_start_func;
        }
    }

    threads::policies::callback_notifier::on_startstop_type
        get_thread_on_stop_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_stop_func();
        }
        else
        {
            return global_on_stop_func;
        }
    }

    threads::policies::callback_notifier::on_error_type
        get_thread_on_error_func()
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_error_func();
        }
        else
        {
            return global_on_error_func;
        }
    }

    threads::policies::callback_notifier::on_startstop_type
    register_thread_on_start_func(
        threads::policies::callback_notifier::on_startstop_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_start_func(std::move(f));
        }

        threads::policies::callback_notifier::on_startstop_type newf =
            std::move(f);
        std::swap(global_on_start_func, newf);
        return newf;
    }

    threads::policies::callback_notifier::on_startstop_type
    register_thread_on_stop_func(
        threads::policies::callback_notifier::on_startstop_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_stop_func(std::move(f));
        }

        threads::policies::callback_notifier::on_startstop_type newf =
            std::move(f);
        std::swap(global_on_stop_func, newf);
        return newf;
    }

    threads::policies::callback_notifier::on_error_type
    register_thread_on_error_func(
        threads::policies::callback_notifier::on_error_type&& f)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->on_error_func(std::move(f));
        }

        threads::policies::callback_notifier::on_error_type newf =
            std::move(f);
        std::swap(global_on_error_func, newf);
        return newf;
    }

    ///////////////////////////////////////////////////////////////////////////
    runtime& get_runtime()
    {
        HPX_ASSERT(get_runtime_ptr() != nullptr);
        return *get_runtime_ptr();
    }

    runtime*& get_runtime_ptr()
    {
        static thread_local runtime* runtime_;
        return runtime_;
    }

    naming::gid_type const & get_locality()
    {
        return get_runtime().get_agas_client().get_local_locality();
    }

    std::string get_thread_name()
    {
        std::string& thread_name = detail::thread_name();
        if (thread_name.empty()) return "<unknown>";
        return thread_name;
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

    void report_error(std::size_t num_thread, std::exception_ptr const& e)
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

    void report_error(std::exception_ptr const& e)
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
        //! FIXME runtime_configuration should probs be a member of
        // hpx::runtime only, not command_line_handling
        //! FIXME change functions in this section accordingly
        if (get_runtime_ptr() != nullptr)
        {
            return get_runtime().get_config().get_entry(key, dflt);
        }
        if (!resource::is_partitioner_valid())
        {
            return dflt;
        }
        return resource::get_partitioner()
            .get_command_line_switches().rtcfg_.get_entry(key, dflt);
    }

    std::string get_config_entry(std::string const& key, std::size_t dflt)
    {
        if (get_runtime_ptr() != nullptr)
        {
            return get_runtime().get_config().get_entry(key, dflt);
        }
        if (!resource::is_partitioner_valid())
        {
            return std::to_string(dflt);
        }
        return resource::get_partitioner()
            .get_command_line_switches().rtcfg_.get_entry(key, dflt);
    }

    // set entries
    void set_config_entry(std::string const& key, std::string const& value)
    {
        if (get_runtime_ptr() != nullptr)
        {
            get_runtime_ptr()->get_config().add_entry(key, value);
            return;
        }
        if (resource::is_partitioner_valid())
        {
            resource::get_partitioner()
                .get_command_line_switches().rtcfg_.add_entry(key, value);
            return;
        }
    }

    void set_config_entry(std::string const& key, std::size_t value)
    {
        if (get_runtime_ptr() != nullptr)
        {
            get_runtime_ptr()->get_config().add_entry(
                key, std::to_string(value));
            return;
        }
        if (resource::is_partitioner_valid())
        {
            resource::get_partitioner()
                .get_command_line_switches().rtcfg_.
                    add_entry(key, std::to_string(value));
            return;
        }
    }

    void set_config_entry_callback(std::string const& key,
        util::function_nonser<void(
            std::string const&, std::string const&)> const& callback)
    {
        if (get_runtime_ptr() != nullptr)
        {
            get_runtime_ptr()->get_config().add_notification_callback(
                key, callback);
            return;
        }
        if (resource::is_partitioner_valid())
        {
            resource::get_partitioner()
                .get_command_line_switches()
                .rtcfg_.add_notification_callback(key, callback);
            return;
        }
    }

    namespace util {
        ///////////////////////////////////////////////////////////////////////////
        // retrieve the command line arguments for the current locality
        bool retrieve_commandline_arguments(
            hpx::program_options::options_description const& app_options,
            hpx::program_options::variables_map& vm)
        {
            // The command line for this application instance is available from
            // this configuration section:
            //
            //     [hpx]
            //     cmd_line=....
            //
            std::string cmdline;
            std::size_t node = std::size_t(-1);

            hpx::util::section& cfg = hpx::get_runtime().get_config();
            if (cfg.has_entry("hpx.cmd_line"))
                cmdline = cfg.get_entry("hpx.cmd_line");
            if (cfg.has_entry("hpx.locality"))
                node = hpx::util::from_string<std::size_t>(
                    cfg.get_entry("hpx.locality"));

            return parse_commandline(
                cfg, app_options, cmdline, vm, node, allow_unregistered);
        }

        ///////////////////////////////////////////////////////////////////////////
        // retrieve the command line arguments for the current locality
        bool retrieve_commandline_arguments(
            std::string const& appname, hpx::program_options::variables_map& vm)
        {
            using hpx::program_options::options_description;

            options_description desc_commandline(
                "Usage: " + appname + " [options]");

            return retrieve_commandline_arguments(desc_commandline, vm);
        }
    }    // namespace util

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

        static naming::id_type here(
            hpx::applier::get_applier().get_raw_locality(ec),
            naming::id_type::unmanaged);
        return here;
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

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
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
#endif

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
        if (std::size_t(-1) != get_worker_thread_num())
            return numa_sensitive;
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
        if (!detail::exit_called)
        {
            runtime* rt = get_runtime_ptr();
            if (nullptr != rt)
                return rt->get_state() == state_stopped;
        }
        return true;        // assume stopped
    }

    bool is_stopped_or_shutting_down()
    {
        runtime* rt = get_runtime_ptr();
        if (!detail::exit_called && nullptr != rt)
        {
            state st = rt->get_state();
            return st >= state_shutdown;
        }
        return true;        // assume stopped
    }

    bool HPX_EXPORT tolerate_node_faults()
    {
#ifdef HPX_HAVE_FAULT_TOLERANCE
        return true;
#else
        return false;
#endif
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
#if defined(HPX_HAVE_NETWORKING)
namespace hpx { namespace parcelset
{
    bool do_background_work(
        std::size_t num_thread, parcelport_background_mode mode)
    {
        return get_runtime().get_parcel_handler().do_background_work(
            num_thread, mode);
    }
}}
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    // shortcut for get_applier().get_thread_manager()
    threadmanager& get_thread_manager()
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

    HPX_API_EXPORT void add_scheduler_mode(threads::policies::scheduler_mode m)
    {
        get_runtime().get_thread_manager().add_scheduler_mode(m);
    }

    HPX_API_EXPORT void add_remove_scheduler_mode(
        threads::policies::scheduler_mode to_add_mode,
        threads::policies::scheduler_mode to_remove_mode)
    {
        get_runtime().get_thread_manager().add_remove_scheduler_mode(
            to_add_mode, to_remove_mode);
    }

    HPX_API_EXPORT void remove_scheduler_mode(threads::policies::scheduler_mode m)
    {
        get_runtime().get_thread_manager().remove_scheduler_mode(m);
    }

    HPX_API_EXPORT topology const& get_topology()
    {
        hpx::runtime* rt = hpx::get_runtime_ptr();
        if (rt == nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::threads::get_topology",
                "the hpx runtime system has not been initialized yet");
        }
        return rt->get_topology();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    std::uint32_t get_locality_id(error_code& ec)
    {
        return agas::get_locality_id(ec);
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

    void reinit_active_counters(bool reset, error_code& ec)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt) {
            rt->reinit_active_counters(reset, ec);
        }
        else {
            HPX_THROWS_IF(ec, invalid_status, "reinit_active_counters",
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

#if defined(HPX_HAVE_NETWORKING)
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
        get_message_handler_registrations().push_back(
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
#endif

    // helper function to stop evaluating counters during shutdown
    void stop_evaluating_counters(bool terminate)
    {
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
            rt->stop_evaluating_counters(terminate);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return true if networking is enabled.
    bool is_networking_enabled()
    {
#if defined(HPX_HAVE_NETWORKING)
        runtime* rt = get_runtime_ptr();
        if (nullptr != rt)
        {
            return rt->get_config().enable_networking();
        }
        return true;        // be on the safe side, enable networking
#else
        return false;
#endif
    }
}
